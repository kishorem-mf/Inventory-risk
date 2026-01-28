# Master Agent v1

This folder contains the **Master Agent v1** implementation, including Gradio UI, pipeline scripts, and supporting modules.

---

## 📂 Files in this folder
- `gradio_ui_master.py` → Main entry point (Gradio UI)  
- `master_agent_ss_1.PNG` → Screenshot  
- `master_agent_v1.ipynb` → Notebook version  
- `reasoning_agent_pipeline.py` → Reasoning pipeline  
- `sql_agent.py` → SQL agent  
- `sql_text_conversion.py` → SQL exceution pipeline

---

## 🚀 How to Run Locally

### 1. Clone only this folder
If you don’t want the full repo, just clone this folder using `svn`:

```bash
svn export https://github.com/Bristlecone-AI-Studio/Inventory-Risk-Explainability/edit/main/Development-Files/Master-Agent/Master_agent_v1
```

This will create a local folder Master_agent_v1 with only the files you see above.

### 2. Create a virtual environment

```bash
cd Master_agent_v1
```

#### Create venv
```bash
python3 -m venv venv
```

#### Activate venv (Linux/macOS)

```bash
source venv/bin/activate
```

#### Activate venv (Windows Git Bash)

```bash
source venv/Scripts/activate
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```
(If requirements.txt is not provided, install manually as needed, e.g. pip install gradio pandas)

### 4. Run the Gradio app

```bash
python gradio_ui_master.py
```

This will start a Gradio interface in your browser (default: http://127.0.0.1:7860/).

<img width="1328" height="524" alt="image" src="https://github.com/user-attachments/assets/d6040fb4-97c3-4b47-89b7-b5080989ac54" />

