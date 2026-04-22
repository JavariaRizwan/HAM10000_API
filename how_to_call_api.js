// ─────────────────────────────────────────────────────────────
// HOW TO CALL THE API FROM ANY WEB PROJECT
// Replace YOUR_API_URL with your deployed URL, e.g.:
//   https://your-app.onrender.com
//   https://your-app-xxxx.run.app
// ─────────────────────────────────────────────────────────────


// ── 1. React / Next.js / Vanilla JS ──────────────────────────
async function predictSkinLesion(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);   // imageFile = File object from <input type="file">

  const response = await fetch("YOUR_API_URL/predict", {
    method: "POST",
    body: formData,
    // Do NOT set Content-Type manually — browser sets it with boundary
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail);
  }

  const result = await response.json();
  /*
  result = {
    "predicted_class": "Melanocytic nevi",
    "confidence": 0.9123,
    "all_probabilities": {
      "Actinic keratoses": 0.0021,
      "Basal cell carcinoma": 0.0045,
      "Benign keratosis-like lesions": 0.0123,
      "Dermatofibroma": 0.0034,
      "Melanocytic nevi": 0.9123,
      "Melanoma": 0.0542,
      "Vascular lesions": 0.0112
    },
    "disclaimer": "This is a research tool, not a medical diagnosis."
  }
  */
  return result;
}

// Example usage in a React component:
//
// const handleUpload = async (e) => {
//   const file = e.target.files[0];
//   const result = await predictSkinLesion(file);
//   console.log(result.predicted_class, result.confidence);
// };


// ── 2. Test with curl (terminal) ─────────────────────────────
//
// curl -X POST "YOUR_API_URL/predict" \
//      -F "file=@/path/to/your/skin_image.jpg"


// ── 3. Python (e.g. another backend or script) ───────────────
//
// import requests
//
// with open("skin_image.jpg", "rb") as f:
//     response = requests.post(
//         "YOUR_API_URL/predict",
//         files={"file": ("skin_image.jpg", f, "image/jpeg")}
//     )
// print(response.json())