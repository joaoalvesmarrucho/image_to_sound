// static/scripts.js

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("conversion-form");
  const progressContainer = document.getElementById("progress-container");
  const progressBar = document.getElementById("progress-bar");
  const outputContainer = document.getElementById("output-container");
  const downloadLink = document.getElementById("download-link");

  form.addEventListener("submit", function (e) {
    e.preventDefault(); // Prevent default form submission

    // Show progress indicator
    progressContainer.style.display = "block";
    progressBar.value = 0;
    outputContainer.style.display = "none";

    const formData = new FormData(form);

    // Send the form data via fetch
    fetch("/convert", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        const taskId = data.task_id;
        const outputFolder = data.output_folder;

        // Poll for progress updates
        const interval = setInterval(() => {
          fetch(`/progress/${taskId}`)
            .then((res) => res.json())
            .then((status) => {
              if (status.status === "invalid") {
                clearInterval(interval);
                alert("Invalid task ID.");
              } else if (status.status === "error") {
                clearInterval(interval);
                alert("An error occurred during processing.");
                progressContainer.style.display = "none";
              } else if (status.status === "processing") {
                progressBar.value = status.progress;
                if (status.progress >= 100) {
                  clearInterval(interval);
                  progressContainer.style.display = "none";
                  // Show download link
                  outputContainer.style.display = "block";
                  const audioPath = `${outputFolder}/output_audio.mp3`;
                  downloadLink.href = `/${audioPath}`;
                }
              }
            })
            .catch((err) => {
              console.error("Error fetching progress:", err);
              clearInterval(interval);
              alert("Failed to fetch progress.");
              progressContainer.style.display = "none";
            });
        }, 1000); // Poll every second
      })
      .catch((err) => {
        console.error("Error submitting form:", err);
        alert("Failed to submit the form.");
        progressContainer.style.display = "none";
      });
  });
});
