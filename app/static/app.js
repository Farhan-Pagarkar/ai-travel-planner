// Date formatting & input initialization helpers
function getLocalDateString(d) {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function initializeDateInputs() {
  const startDateInput = document.getElementById("start_date");
  const endDateInput = document.getElementById("end_date");
  if (!startDateInput || !endDateInput) return;

  const now = new Date();
  const todayStr = getLocalDateString(now);

  startDateInput.min = todayStr;
  endDateInput.min = todayStr;

  // Initialize with sensible future dates if empty: start in 7 days, 4-day trip
  if (!startDateInput.value) {
    const defaultStart = new Date(now);
    defaultStart.setDate(defaultStart.getDate() + 7);
    startDateInput.value = getLocalDateString(defaultStart);

    const defaultEnd = new Date(defaultStart);
    defaultEnd.setDate(defaultEnd.getDate() + 3);
    endDateInput.value = getLocalDateString(defaultEnd);
    endDateInput.min = startDateInput.value;
  }

  startDateInput.addEventListener("change", () => {
    if (startDateInput.value) {
      endDateInput.min = startDateInput.value;
      if (endDateInput.value && endDateInput.value < startDateInput.value) {
        endDateInput.value = startDateInput.value;
      }
    }
  });
}

// AI Travel Planner - Frontend Application Logic

let currentPlanId = null;
let currentPlanData = null;

// DOM Elements
const planForm = document.getElementById("plan-form");
const submitBtn = document.getElementById("submit-btn");
const formError = document.getElementById("form-error");
const loadingIndicator = document.getElementById("loading-indicator");
const loadingText = document.getElementById("loading-text");
const globalError = document.getElementById("global-error");
const planSection = document.getElementById("plan-section");

const displayPlanId = document.getElementById("display-plan-id");
const statusBadge = document.getElementById("status-badge");
const displayRevisionCount = document.getElementById("display-revision-count");

const validatorBox = document.getElementById("validator-box");
const validatorIssuesList = document.getElementById("validator-issues-list");
const warningsBox = document.getElementById("warnings-box");
const warningsList = document.getElementById("warnings-list");

const itineraryTitle = document.getElementById("itinerary-title");
const totalCostBadge = document.getElementById("total-cost-badge");
const itinerarySummary = document.getElementById("itinerary-summary");
const budgetBreakdown = document.getElementById("budget-breakdown");
const daysContainer = document.getElementById("days-container");

const reviewControls = document.getElementById("review-controls");
const btnApprove = document.getElementById("btn-approve");
const btnReject = document.getElementById("btn-reject");
const btnModify = document.getElementById("btn-modify");
const rejectFeedback = document.getElementById("reject-feedback");
const modifyInstructions = document.getElementById("modify-instructions");

const finalizedBox = document.getElementById("finalized-box");
const btnViewFinal = document.getElementById("btn-view-final");
const revisionLimitBox = document.getElementById("revision-limit-box");

// Helper: Escape HTML to avoid injection
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Helper: Format error message from API response
function formatApiError(data, status) {
  if (!data) {
    return `Server error (${status}). Please try again.`;
  }
  if (typeof data.detail === "string") {
    return data.detail;
  }
  if (Array.isArray(data.detail)) {
    // Pydantic validation error list
    return data.detail
      .map(err => {
        const field = err.loc ? err.loc.filter(l => l !== "body").join(".") : "field";
        return `${field ? field + ": " : ""}${err.msg}`;
      })
      .join("; ");
  }
  return data.message || `An error occurred (${status}).`;
}

// Show/Hide Helpers
function showError(el, message) {
  el.textContent = message;
  el.classList.remove("hidden");
}

function hideError(el) {
  el.textContent = "";
  el.classList.add("hidden");
}

function setLoading(isLoading, text = "Processing...") {
  if (isLoading) {
    loadingText.textContent = text;
    loadingIndicator.classList.remove("hidden");
    submitBtn.disabled = true;
    if (btnApprove) btnApprove.disabled = true;
    if (btnReject) btnReject.disabled = true;
    if (btnModify) btnModify.disabled = true;
  } else {
    loadingIndicator.classList.add("hidden");
    submitBtn.disabled = false;
    if (btnApprove) btnApprove.disabled = false;
    if (btnReject) btnReject.disabled = false;
    if (btnModify) btnModify.disabled = false;
  }
}

// Form Submission: Create Plan
planForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError(formError);
  hideError(globalError);

  const destination = document.getElementById("destination").value.trim();
  const startDate = document.getElementById("start_date").value;
  const endDate = document.getElementById("end_date").value;
  const rawBudget = document.getElementById("total_trip_budget").value;
  const currency = document.getElementById("currency").value.trim().toUpperCase();
  const rawTravelers = document.getElementById("traveler_count").value;
  const rawInterests = document.getElementById("interests").value.trim();
  const pace = document.getElementById("pace").value;
  const preferenceNotes = document.getElementById("preference_notes").value.trim();

  // Client-side validations
  if (!destination) {
    showError(formError, "Destination is required.");
    return;
  }
  if (!startDate || !endDate) {
    showError(formError, "Start and end dates are required.");
    return;
  }
  const todayStr = getLocalDateString(new Date());
  if (startDate < todayStr) {
    showError(formError, `Start date (${startDate}) cannot be in the past. Please select a date on or after today (${todayStr}).`);
    return;
  }
  if (endDate < startDate) {
    showError(formError, "End date cannot precede start date.");
    return;
  }
  const sParts = startDate.split("-").map(Number);
  const eParts = endDate.split("-").map(Number);
  const sDate = new Date(sParts[0], sParts[1] - 1, sParts[2]);
  const eDate = new Date(eParts[0], eParts[1] - 1, eParts[2]);
  const durationDays = Math.round((eDate - sDate) / (1000 * 60 * 60 * 24)) + 1;
  if (durationDays > 14) {
    showError(formError, `Trip duration (${durationDays} days) exceeds the maximum allowed limit of 14 calendar days inclusive.`);
    return;
  }
  const budget = parseFloat(rawBudget);
  if (isNaN(budget) || budget <= 0) {
    showError(formError, "Total budget must be a positive number.");
    return;
  }
  const travelers = parseInt(rawTravelers, 10);
  if (isNaN(travelers) || travelers < 1) {
    showError(formError, "Traveler count must be at least 1.");
    return;
  }
  const interests = rawInterests
    .split(",")
    .map(i => i.trim())
    .filter(i => i.length > 0);
  if (interests.length === 0) {
    showError(formError, "Please enter at least one interest.");
    return;
  }

  // Build exact API payload
  const payload = {
    destination: destination,
    start_date: startDate,
    end_date: endDate,
    total_trip_budget: budget,
    currency: currency,
    interests: interests,
    traveler_count: travelers,
    preferences: {
      pace: pace || "moderate",
      ...(preferenceNotes ? { notes: preferenceNotes } : {}),
    },
  };

  setLoading(true, "Researching destination, synthesizing itinerary, and validating constraints...");

  try {
    const res = await fetch("/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json().catch(() => null);

    if (!res.ok) {
      showError(formError, formatApiError(data, res.status));
      return;
    }

    currentPlanId = data.plan_id;
    currentPlanData = data;
    renderPlan(data);
    planSection.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError(formError, `Network error: ${err.message}. Please check your connection.`);
  } finally {
    setLoading(false);
  }
});

// Render the Plan state and itinerary
function renderPlan(data) {
  if (!data) return;

  planSection.classList.remove("hidden");

  // Metadata & Status
  displayPlanId.textContent = data.plan_id || "-";
  const status = data.status || "unknown";
  statusBadge.textContent = status.replace(/_/g, " ");
  statusBadge.className = `badge badge-${status}`;
  displayRevisionCount.textContent = data.revision_count !== undefined ? data.revision_count : 0;

  // Validator Findings
  const issues = data.validation_issues || [];
  if (issues.length > 0) {
    validatorBox.classList.remove("hidden");
    validatorIssuesList.innerHTML = issues
      .map(issue => `<li>${escapeHtml(issue)}</li>`)
      .join("");
  } else {
    validatorBox.classList.add("hidden");
    validatorIssuesList.innerHTML = "";
  }

  // Research Warnings & Caveats
  const warnings = [];
  const itin = data.draft_itinerary || data.final_itinerary || {};
  if (itin.warnings && Array.isArray(itin.warnings)) {
    warnings.push(...itin.warnings);
  }
  if (data.research_results && Array.isArray(data.research_results.warnings_and_uncertainties)) {
    warnings.push(...data.research_results.warnings_and_uncertainties);
  }
  const uniqueWarnings = Array.from(new Set(warnings)).filter(w => Boolean(w && w.trim()));
  if (uniqueWarnings.length > 0) {
    warningsBox.classList.remove("hidden");
    warningsList.innerHTML = uniqueWarnings
      .map(w => `<li>${escapeHtml(w)}</li>`)
      .join("");
  } else {
    warningsBox.classList.add("hidden");
    warningsList.innerHTML = "";
  }

  // Itinerary Header & Cost
  const isFinal = Boolean(data.final_itinerary || status === "finalized");
  itineraryTitle.textContent = isFinal
    ? `Finalized Itinerary: ${itin.destination || ""}`
    : `Draft Itinerary: ${itin.destination || ""}`;

  const curr = itin.currency || "USD";
  const totalCost = itin.total_estimated_cost !== undefined ? itin.total_estimated_cost : 0.0;
  totalCostBadge.textContent = `Total Estimated Cost: ${totalCost.toFixed(2)} ${curr} (Estimate)`;

  itinerarySummary.textContent = itin.summary || "No summary provided.";

  // Budget Breakdown if present
  if (itin.budget_breakdown && itin.budget_breakdown.breakdown) {
    budgetBreakdown.classList.remove("hidden");
    const b = itin.budget_breakdown;
    const catItems = Object.entries(b.breakdown)
      .map(([cat, amount]) => `<li><strong>${escapeHtml(cat)}:</strong> ${amount} ${escapeHtml(b.currency || curr)} (Estimate)</li>`)
      .join("");
    budgetBreakdown.innerHTML = `
      <strong>Budget Allocation Estimates:</strong>
      <ul>${catItems}</ul>
    `;
  } else {
    budgetBreakdown.classList.add("hidden");
    budgetBreakdown.innerHTML = "";
  }

  // Render Days
  const days = itin.days || [];
  if (days.length === 0) {
    daysContainer.innerHTML = "<p>No daily activities scheduled.</p>";
  } else {
    daysContainer.innerHTML = days.map(renderDayCard).join("");
  }

  // Workflow Stage Controls
  if (status === "awaiting_review") {
    reviewControls.classList.remove("hidden");
    finalizedBox.classList.add("hidden");
    revisionLimitBox.classList.add("hidden");
  } else if (status === "finalized") {
    reviewControls.classList.add("hidden");
    finalizedBox.classList.remove("hidden");
    revisionLimitBox.classList.add("hidden");
  } else if (status === "revision_limit_reached") {
    reviewControls.classList.add("hidden");
    finalizedBox.classList.add("hidden");
    revisionLimitBox.classList.remove("hidden");
  } else {
    reviewControls.classList.add("hidden");
    finalizedBox.classList.add("hidden");
    revisionLimitBox.classList.add("hidden");
  }
}

// Render an individual Day Card
function renderDayCard(day) {
  const dayNum = day.day || 1;
  const dayDate = day.date || "";
  const focus = day.focus ? `— ${escapeHtml(day.focus)}` : "";
  const dayCost = day.estimated_cost !== undefined
    ? `Estimated Day Cost: ${Number(day.estimated_cost).toFixed(2)} (Estimate)`
    : "";

  const activities = day.activities || [];
  const activitiesHtml = activities.length > 0
    ? activities.map(renderActivityItem).join("")
    : "<p class='field-hint'>No activities planned for this day.</p>";

  return `
    <div class="day-card">
      <div class="day-header">
        <span class="day-title">Day ${dayNum} (${escapeHtml(dayDate)}) <span class="day-focus">${focus}</span></span>
        <span class="day-cost">${escapeHtml(dayCost)}</span>
      </div>
      <div class="activities-list">
        ${activitiesHtml}
      </div>
    </div>
  `;
}

// Render an Activity Item
function renderActivityItem(act) {
  const name = escapeHtml(act.name || "Activity");
  const desc = escapeHtml(act.description || "");
  const category = escapeHtml(act.category || "activity");
  const time = act.start_time ? escapeHtml(act.start_time) : "";
  const duration = act.estimated_duration ? escapeHtml(act.estimated_duration) : "";
  const metaParts = [category, time, duration].filter(Boolean).join(" • ");

  const cost = act.estimated_cost !== undefined
    ? `Estimated Cost: ${Number(act.estimated_cost).toFixed(2)} (Estimate)`
    : "Estimated Cost: 0.00 (Estimate)";

  // Verified Source URL & Candidate Provenance
  let provenanceHtml = "";
  if (act.source_url) {
    const safeUrl = escapeHtml(act.source_url);
    const candIdText = act.candidate_id ? `[Verified: ${escapeHtml(act.candidate_id)}] ` : "";
    provenanceHtml = `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer" class="provenance-link">${candIdText}Verified Research Source</a>`;
  } else {
    provenanceHtml = `<span class="provenance-generic">Generic Schedule Block (Unverified)</span>`;
  }

  return `
    <div class="activity-item">
      <div class="activity-top">
        <span class="activity-name">${name}</span>
        <span class="activity-meta">${metaParts}</span>
      </div>
      <p class="activity-desc">${desc}</p>
      <div class="activity-footer">
        <span>${cost}</span>
        <span>${provenanceHtml}</span>
      </div>
    </div>
  `;
}

// Review Action: Submit Decision (Approve, Reject, Modify)
async function submitReview(actionPayload, actionDescription) {
  if (!currentPlanId) {
    showError(globalError, "No active plan found to review.");
    return;
  }

  hideError(globalError);
  setLoading(true, actionDescription);

  try {
    const res = await fetch(`/plan/${encodeURIComponent(currentPlanId)}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(actionPayload),
    });

    const data = await res.json().catch(() => null);

    if (!res.ok) {
      showError(globalError, formatApiError(data, res.status));
      return;
    }

    currentPlanData = data;
    renderPlan(data);

    // Clear feedback/modifications textareas
    if (rejectFeedback) rejectFeedback.value = "";
    if (modifyInstructions) modifyInstructions.value = "";

    // If approved and finalized, fetch final itinerary
    if (data.status === "finalized") {
      await fetchFinalPlan(currentPlanId);
    }
  } catch (err) {
    showError(globalError, `Network error during review: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// Approve Handler
btnApprove.addEventListener("click", () => {
  submitReview({ action: "approve" }, "Finalizing and approving travel itinerary...");
});

// Reject Handler
btnReject.addEventListener("click", () => {
  const feedback = rejectFeedback.value.trim();
  if (!feedback) {
    showError(globalError, "Feedback is required when rejecting a plan for revision.");
    rejectFeedback.focus();
    return;
  }
  submitReview({ action: "reject", feedback: feedback }, "Revising itinerary based on feedback...");
});

// Modify Handler
btnModify.addEventListener("click", () => {
  const instructions = modifyInstructions.value.trim();
  if (!instructions) {
    showError(globalError, "Specific modification instructions are required.");
    modifyInstructions.focus();
    return;
  }
  submitReview({ action: "modify", modifications: instructions }, "Incorporating requested modifications...");
});

// Fetch Final Approved Plan Handler
btnViewFinal.addEventListener("click", () => {
  if (currentPlanId) {
    fetchFinalPlan(currentPlanId);
  }
});

async function fetchFinalPlan(planId) {
  hideError(globalError);
  setLoading(true, "Fetching finalized itinerary...");

  try {
    const res = await fetch(`/plan/${encodeURIComponent(planId)}/final`);
    const data = await res.json().catch(() => null);

    if (!res.ok) {
      showError(globalError, formatApiError(data, res.status));
      return;
    }

    // Merge final itinerary into current plan view
    currentPlanData = {
      ...currentPlanData,
      status: "finalized",
      final_itinerary: data.final_itinerary,
    };
    renderPlan(currentPlanData);
  } catch (err) {
    showError(globalError, `Error retrieving final plan: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// Initialize date inputs on load
initializeDateInputs();
