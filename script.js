async function analyzeSkin() {
  const fileInput = document.getElementById("imageInput");

  if (!fileInput.files.length) {
    alert("Please upload an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    console.log(data);

    document.getElementById("result").innerHTML = `
      <h3>Skin Report</h3>

      <b>Prediction:</b> ${data.prediction || "N/A"} <br>
      <b>Confidence:</b> ${data.confidence ? data.confidence.toFixed(2) : "N/A"}% <br><br>

      Acne: ${data.acne || 0}/10 <br>
      Pigmentation: ${data.pigmentation || 0}/10 <br>
      Oiliness: ${data.oiliness || 0}/10 <br>

      <br>
      <b>Overall Score:</b> ${data.overall || 0}

      <br><br>
      <b>AI Summary:</b> ${data.summary || "No summary"}

      <br><br>
      <b>Recommendations:</b>
      <ul>
        ${(data.recommendations || []).map(r => `<li>${r}</li>`).join("")}
      </ul>
    `;

  } catch (error) {
    console.error("Prediction error:", error);
    alert("Prediction failed. Make sure Flask server is running.");
  }
}