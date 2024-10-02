document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('input-form');
    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the form from submitting traditionally

        if (!validateForm()) {
            return false; // Stop the function if the form is not valid
        }

        // Collect form data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = parseFloat(value); // Convert input values to float
            displayLevel(key, value); // Evaluate and display each input level
        });

        // Make the fetch API POST request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('prediction-result').textContent = 'Failed to get prediction: ' + error.message;
        });
    });
});

function displayResults(data) {
    let resultText = "<strong>Prediction:</strong> No, readmission not likely.";
    if (data.prediction === 1) {
        const probabilityPercent = (data.probability * 100).toFixed(2); // Convert to percentage
        resultText = `<strong>Prediction:</strong> Yes, Possible readmission in ICU.<br><strong>Probability:</strong> ${probabilityPercent}%`;
    }
    document.getElementById('prediction-result').innerHTML = resultText;
}

function displayLevel(key, value) {
    const ranges = {
        hematocrit: { min: 38.3, max: 48.6 },
        neutrophils: { min: 1.5, max: 8.0 },
        sodium: { min: 135, max: 145 },
        glucose: { min: 70, max: 99 },
        bloodureanitro: { min: 6, max: 20 },
        creatinine: { min: 0.74, max: 1.35 },
        bmi: { min: 18.5, max: 24.9 },
        pulse: { min: 60, max: 100 },
        respiration: { min: 12, max: 16 },
        lengthofstay: { min: 1, max: 30 } // Display no feedback for length of stay
    };

    const feedbackElement = document.getElementById(`${key}-level`);
    if (!ranges[key] || key === 'lengthofstay') {
        feedbackElement.textContent = '';
        return;
    }

    let numValue = parseFloat(value);
    let rangeText = `(${ranges[key].min} - ${ranges[key].max})`;
    if (numValue < ranges[key].min) {
        feedbackElement.textContent = `Below normal level ${rangeText}`;
    } else if (numValue > ranges[key].max) {
        feedbackElement.textContent = `Above normal level ${rangeText}`;
    } else {
        feedbackElement.textContent = `Normal level ${rangeText}`;
    }
}

function validateForm() {
    let isValid = true;
    const inputs = document.querySelectorAll('#input-form input[type="number"]');
    inputs.forEach(input => {
        const feedbackElement = input.nextElementSibling;
        feedbackElement.textContent = ''; // Clear previous feedback
        if (!input.value.trim()) {
            feedbackElement.textContent = 'This field is required';
            input.classList.add('error'); // Add 'error' class to highlight
            isValid = false;
        } else {
            input.classList.remove('error'); // Remove 'error' class if filled
        }
    });

    return isValid;
}