<!DOCTYPE html>
<html>
<head>
  <title>Diabetes Prediction</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background-color: #f9f9f9;
      padding: 20px;
    }
    
    .container {
      max-width: 500px;  
      margin: 0 auto;
      background-color: #fff;
      padding: 20px; 
      border-radius: 5px;
    }
    
    h1 {
      text-align: center; 
    }
    
    label {
      display: block;
      margin-bottom: 10px;
    }
    
    input[type="text"], 
    input[type="number"] {
      padding: 10px;
      border-radius: 5px;
      border: 1px solid #ccc;  
      width: 100%;
      margin-bottom: 20px; 
    }
    
    button {
      background-color: #4CAF50;
      color: white;
      padding: 12px 20px;
      border: none;
      border-radius: 5px;
      cursor: pointer; 
      width: 100%;
      font-size: 16px;
    }

  </style>
</head>

<body>

  <div class="container">

    <h1>Diabetes Prediction</h1>
    
    <form>
    
      <label for="gender">Gender:</label>
      <input type="text" id="gender" name="gender">

      <label for="age">Age:</label>    
      <input type="number" id="age" name="age">
      
      <label for="hypertension">Hypertension:</label>    
      <input type="number" id="hypertension" name="hypertension">

      <label for="heart_disease">Heart Disease:</label>
      <input type="number" id="heart_disease" name="heart_disease">
      
      <label for="smoking_history">Smoking History:</label>    
      <input type="number" id="smoking_history" name="smoking_history">

      <label for="bmi">BMI:</label>
      <input type="number" step="0.01" id="bmi" name="bmi">
      
      <label for="hba1c_level">HbA1c Level:</label>
      <input type="number" step="0.01" id="hba1c_level" name="hba1c_level">

      <label for="blood_glucose_level">Blood Glucose Level:</label>    
      <input type="number" id="blood_glucose_level" name="blood_glucose_level">
      
      <button type="button" onclick="predict()">Predict</button>
      
    </form>

  </div>

  <script>
    function predict() {
      // Get input values  
      var gender = document.getElementById("gender").value;  
      var age = document.getElementById("age").value;
      var hypertension = document.getElementById("hypertension").value;
      var heart_disease = document.getElementById("heart_disease").value;  
      var smoking_history = document.getElementById("smoking_history").value;
      var bmi = document.getElementById("bmi").value;
      var hba1c_level = document.getElementById("hba1c_level").value;
      var blood_glucose_level = document.getElementById("blood_glucose_level").value;

      // API URL
      var apiUrl = "REPLACE_WITH_PUBLIC_API_URL/api/v1.0/predict/"+gender+"/"+age+"/"+hypertension+"/"+heart_disease+"/"+smoking_history+"/"+bmi+"/"+hba1c_level+"/"+blood_glucose_level;
      
      // Make API call  
      fetch(apiUrl)
        .then(response => response.json())
        .then(data => {
          // Display results  
          console.log(data)
          alert("Prediction: " + data.prediction + ", Probability: " + data.probability[1]); 
        });
    }
  </script>

</body>
</html>