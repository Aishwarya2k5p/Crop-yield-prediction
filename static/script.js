document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictionForm");
  const resultEl = document.getElementById("result");
  const submitBtn = form.querySelector("button[type='submit']");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    // Reset result and show loading state
    resultEl.textContent = "";
    resultEl.style.opacity = 0;
    submitBtn.disabled = true;
    submitBtn.textContent = "Predicting...";

    const formData = new FormData(e.target);
    const jsonData = Object.fromEntries(formData.entries());

    // Convert numeric inputs
    for (let key in jsonData) {
      if (!isNaN(jsonData[key])) {
        jsonData[key] = Number(jsonData[key]);
      }
    }

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(jsonData),
      });

      const result = await response.json();

      // Show result with fade-in
      if (result.predicted_yield !== undefined) {
        resultEl.textContent = "🌾 Predicted Yield: " + result.predicted_yield + " tons/hectare";
        resultEl.style.color = "#2f855a";
      } else {
        resultEl.textContent = "❌ Error: " + (result.error || "Unknown error occurred");
        resultEl.style.color = "red";
      }

      resultEl.style.opacity = 1;
      resultEl.style.transition = "opacity 0.8s ease";
      resultEl.scrollIntoView({ behavior: "smooth" });

    } catch (error) {
      resultEl.textContent = "❌ Network Error: Unable to connect to server.";
      resultEl.style.color = "red";
      resultEl.style.opacity = 1;
    }

    // Reset button state
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict";
  });

  // Optional: highlight empty required fields on blur
  const inputs = form.querySelectorAll("input, select");
  inputs.forEach((input) => {
    input.addEventListener("blur", () => {
      if (input.hasAttribute("required") && !input.value.trim()) {
        input.style.borderColor = "red";
      } else {
        input.style.borderColor = "#ccc";
      }
    });
  });
});
