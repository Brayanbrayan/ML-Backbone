import os

readme_template = """# {project_name}

{description}

## 📊 Dataset
Source and description of the data used.

## 🎯 Objective
Problem being solved and approach taken.

## 🛠️ Tech Stack
- Python 3.10
- Key libraries used in this project

## 📈 Key Findings
Main results and insights.

## 🚀 Usage
See notebook for detailed analysis.

## 📁 Files
- Main notebook with full analysis
- Supporting scripts and data
"""

projects = {
    "Clustering": "Music analysis using clustering",
    "Cuisine": "Multi-class cuisine prediction from ingredients using ONNX",
    "NLP": "Sentiment analysis and text processing projects",
    "Reinforcement": "Q-learning and reinforcement learning implementations",
    "UFOs": "UFO sightings analysis with ML predictions and web interface",
    "Timeseries": "Time series forecasting using ARIMA and SVR",
    "pumpkins": "Pumpkin price prediction using regression",
}

base_path = r"E:\bryan - Copy\4.2\dataScience\ML"

for folder, desc in projects.items():
    folder_path = os.path.join(base_path, folder)
    if os.path.exists(folder_path):
        readme_path = os.path.join(folder_path, "README.md")
        content = readme_template.format(
            project_name=folder.replace("-", " ").title(),
            description=desc
        )
        # CRITICAL FIX: encoding='utf-8'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created README for {folder}")
    else:
        print(f"❌ Folder not found: {folder_path}")

print("\n🎉 Done! READMEs created successfully.")