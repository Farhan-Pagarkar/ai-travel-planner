from datetime import date, datetime, timedelta
from typing import Any, Optional
import httpx

from app.schemas.research import (
    DailyWeatherForecast,
    DestinationCoordinates,
    HistoricalWeatherComparison,
    WeatherContext,
)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# WMO Weather Interpretation Codes (WW)
WMO_CODE_MAP: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _interpret_wmo_code(code: Optional[int]) -> str:
    if code is None:
        return "Unknown"
    return WMO_CODE_MAP.get(int(code), f"Weather code {code}")


def _get_equivalent_date(original_date: date, target_year: int) -> date:
    """Safely map a date to a prior year, handling leap days."""
    try:
        return original_date.replace(year=target_year)
    except ValueError:
        # Handles Feb 29 mapped to non-leap year
        return date(target_year, 2, 28)


def _parse_date(date_val: Any) -> date:
    if isinstance(date_val, date):
        return date_val
    return datetime.strptime(str(date_val), "%Y-%m-%d").date()


class WeatherClient:
    """Client for Open-Meteo geocoding, 16-day forecasts, and historical climate archives."""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def geocode_destination(self, destination: str) -> Optional[DestinationCoordinates]:
        """Resolve a destination string into latitude, longitude, and timezone."""
        params = {
            "name": destination.strip(),
            "count": 1,
            "language": "en",
            "format": "json",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(GEOCODING_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception:
            return None

        results = data.get("results")
        if not results or not isinstance(results, list):
            return None

        first = results[0]
        try:
            lat = float(first["latitude"])
            lon = float(first["longitude"])
            name = str(first.get("name", destination))
            country = first.get("country")
            resolved_name = f"{name}, {country}" if country else name
            tz = str(first.get("timezone", "UTC"))
            return DestinationCoordinates(
                latitude=lat,
                longitude=lon,
                resolved_name=resolved_name,
                timezone=tz,
            )
        except (KeyError, ValueError, TypeError):
            return None

    def get_forecast(
        self,
        lat: float,
        lon: float,
        tz: str,
        target_dates: set[str],
    ) -> list[DailyWeatherForecast]:
        """Fetch 16-day daily forecast and filter to relevant target dates."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
                "weather_code",
            ],
            "timezone": tz,
            "forecast_days": 16,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(FORECAST_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []

        daily = data.get("daily", {})
        times = daily.get("time", [])
        temp_maxs = daily.get("temperature_2m_max", [])
        temp_mins = daily.get("temperature_2m_min", [])
        precips = daily.get("precipitation_sum", [])
        precip_probs = daily.get("precipitation_probability_max", [])
        codes = daily.get("weather_code", [])

        forecast_list: list[DailyWeatherForecast] = []
        for idx, t in enumerate(times):
            if t in target_dates:
                try:
                    prob = (
                        float(precip_probs[idx])
                        if precip_probs and idx < len(precip_probs) and precip_probs[idx] is not None
                        else None
                    )
                    forecast_list.append(
                        DailyWeatherForecast(
                            date=t,
                            temp_max_c=round(float(temp_maxs[idx]), 1),
                            temp_min_c=round(float(temp_mins[idx]), 1),
                            precipitation_sum_mm=round(float(precips[idx]), 1),
                            precipitation_probability_pct=prob,
                            weather_condition=_interpret_wmo_code(codes[idx] if idx < len(codes) else None),
                        )
                    )
                except (IndexError, ValueError, TypeError):
                    continue

        return forecast_list

    def get_historical_comparison(
        self,
        lat: float,
        lon: float,
        tz: str,
        start_date: date,
        end_date: date,
        years_back: int = 3,
    ) -> Optional[HistoricalWeatherComparison]:
        """Compare corresponding calendar dates across prior years (e.g., past 3 years)."""
        current_year = start_date.year
        valid_years: list[int] = []
        all_max_temps: list[float] = []
        all_min_temps: list[float] = []
        total_precips_by_year: list[float] = []
        rainy_days_by_year: list[int] = []

        last_year_dates_str: Optional[str] = None
        last_year_details: Optional[str] = None

        for offset in range(1, years_back + 1):
            prior_year = start_date.year - offset
            hist_start = _get_equivalent_date(start_date, start_date.year - offset)
            hist_end = _get_equivalent_date(end_date, end_date.year - offset)

            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": hist_start.isoformat(),
                "end_date": hist_end.isoformat(),
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "weather_code",
                ],
                "timezone": tz,
            }

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(ARCHIVE_URL, params=params)
                    if response.status_code != 200:
                        continue
                    data = response.json()
            except Exception:
                continue

            daily = data.get("daily", {})
            maxs = [float(x) for x in daily.get("temperature_2m_max", []) if x is not None]
            mins = [float(x) for x in daily.get("temperature_2m_min", []) if x is not None]
            precips = [float(x) for x in daily.get("precipitation_sum", []) if x is not None]

            if maxs and mins:
                valid_years.append(prior_year)
                all_max_temps.extend(maxs)
                all_min_temps.extend(mins)
                year_precip = sum(precips) if precips else 0.0
                year_rain_days = sum(1 for p in precips if p >= 1.0)
                total_precips_by_year.append(year_precip)
                rainy_days_by_year.append(year_rain_days)

                if offset == 1:
                    last_year_dates_str = f"{hist_start.isoformat()} to {hist_end.isoformat()}"
                    avg_max_last = round(sum(maxs) / len(maxs), 1)
                    avg_min_last = round(sum(mins) / len(mins), 1)
                    year_label = (
                        f"{hist_start.year}-{hist_end.year}"
                        if hist_start.year != hist_end.year
                        else f"{hist_start.year}"
                    )
                    last_year_details = (
                        f"Last year ({year_label}): highs averaged {avg_max_last}°C, "
                        f"lows averaged {avg_min_last}°C, total rainfall was {round(year_precip, 1)}mm."
                    )

        if not valid_years:
            return None

        avg_max = round(sum(all_max_temps) / len(all_max_temps), 1)
        avg_min = round(sum(all_min_temps) / len(all_min_temps), 1)
        avg_precip = (
            round(sum(total_precips_by_year) / len(total_precips_by_year), 1)
            if total_precips_by_year
            else 0.0
        )
        avg_rain_days = (
            round(sum(rainy_days_by_year) / len(rainy_days_by_year))
            if rainy_days_by_year
            else 0
        )

        summary = (
            f"Historical climate comparison across {len(valid_years)} prior years ({', '.join(str(y) for y in valid_years)}): "
            f"typical daily highs around {avg_max}°C and lows around {avg_min}°C. "
            f"Average precipitation over this period was {avg_precip}mm with ~{avg_rain_days} rain day(s). "
        )
        if last_year_details:
            summary += last_year_details

        return HistoricalWeatherComparison(
            comparison_years=valid_years,
            equivalent_dates_last_year=last_year_dates_str,
            typical_temp_max_c=avg_max,
            typical_temp_min_c=avg_min,
            typical_precipitation_mm=avg_precip,
            rainy_days_count=avg_rain_days,
            summary=summary,
        )


def gather_weather_intelligence(
    destination: str,
    start_date_str: str,
    end_date_str: str,
    reference_today: Optional[date] = None,
    client: Optional[WeatherClient] = None,
) -> tuple[Optional[DestinationCoordinates], WeatherContext]:
    """Retrieve weather intelligence distinguishing forecast vs historical data.

    Returns a tuple of (DestinationCoordinates or None, WeatherContext).
    """
    if client is None:
        client = WeatherClient()

    coords = client.geocode_destination(destination)
    if not coords:
        return None, WeatherContext(
            weather_type="unavailable",
            notes=f"Geocoding service was unable to resolve destination '{destination}' for weather data.",
        )

    try:
        start_dt = _parse_date(start_date_str)
        end_dt = _parse_date(end_date_str)
    except Exception:
        return coords, WeatherContext(
            weather_type="unavailable",
            notes="Invalid start or end date format provided for weather lookup.",
        )

    today = reference_today or date.today()
    max_forecast_date = today + timedelta(days=15)

    # Determine date categories
    curr = start_dt
    trip_dates: list[date] = []
    while curr <= end_dt:
        trip_dates.append(curr)
        curr += timedelta(days=1)

    forecast_eligible_dates = {d.isoformat() for d in trip_dates if today <= d <= max_forecast_date}
    beyond_forecast_dates = [d for d in trip_dates if d > max_forecast_date]

    # Scenario 1: All dates covered by 16-day forecast
    if forecast_eligible_dates and not beyond_forecast_dates:
        forecast_days = client.get_forecast(coords.latitude, coords.longitude, coords.timezone, forecast_eligible_dates)
        if forecast_days:
            return coords, WeatherContext(
                weather_type="forecast",
                forecast_days=forecast_days,
                notes="Daily forecast data available directly from Open-Meteo for all requested travel dates.",
            )

    # Scenario 2: All dates beyond 16-day forecast horizon
    if beyond_forecast_dates and not forecast_eligible_dates:
        hist_comp = client.get_historical_comparison(coords.latitude, coords.longitude, coords.timezone, start_dt, end_dt)
        return coords, WeatherContext(
            weather_type="historical_comparison",
            historical_comparison=hist_comp,
            notes=(
                "Travel dates fall beyond the standard 16-day forecast horizon. "
                "Provided data reflects historical climate metrics over equivalent dates from the previous 3 years."
            ),
        )

    # Scenario 3: All dates in the past (historical dates)
    if not forecast_eligible_dates and not beyond_forecast_dates:
        hist_comp = client.get_historical_comparison(coords.latitude, coords.longitude, coords.timezone, start_dt, end_dt)
        return coords, WeatherContext(
            weather_type="historical_comparison",
            historical_comparison=hist_comp,
            notes=(
                "Travel dates are historical past dates. "
                "Provided data reflects historical climate metrics over equivalent dates from the previous 3 years."
            ),
        )

    # Scenario 4: Hybrid (partial forecast overlap)
    # Forecast covers dates within horizon; historical comparison covers ONLY dates beyond horizon
    forecast_days = client.get_forecast(coords.latitude, coords.longitude, coords.timezone, forecast_eligible_dates) if forecast_eligible_dates else []

    covered_dates = sorted(list(forecast_eligible_dates))
    beyond_dates = sorted(beyond_forecast_dates)
    hist_comp = None
    hist_range_str = "none"

    if beyond_dates:
        hist_start_dt = beyond_dates[0]
        hist_end_dt = beyond_dates[-1]
        hist_comp = client.get_historical_comparison(
            coords.latitude, coords.longitude, coords.timezone, hist_start_dt, hist_end_dt
        )
        hist_range_str = f"{hist_start_dt.isoformat()} to {hist_end_dt.isoformat()}"

    forecast_range_str = f"{covered_dates[0]} to {covered_dates[-1]}" if covered_dates else "none"

    return coords, WeatherContext(
        weather_type="hybrid",
        forecast_days=forecast_days,
        historical_comparison=hist_comp,
        notes=(
            f"Trip partially overlaps the 16-day forecast window. "
            f"Live forecast data is provided for {forecast_range_str} ({len(forecast_days)} day(s)); "
            f"historical climate comparisons are provided for the remaining dates beyond the forecast horizon from {hist_range_str}."
        ),
    )
