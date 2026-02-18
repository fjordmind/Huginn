# Huginn
A lightweight, fully local ReAct AI agent powered by Ollama. It uses watchdog to monitor a local file for tasks, autonomously researches via DuckDuckGo, and emails you the final solution."
A well-structured `README.md` is the absolute best way to get traction on GitHub. Since developers love to know exactly *what* a tool does, *what* hardware it needs, and *how* to run it immediately, I have structured this draft to hit all those points.

**Huginn** is a lightweight, fully autonomous, and 100% local AI agent. It runs quietly in the background on your machine, waiting for you to add a task to a text file. Once triggered, it uses a local LLM to reason, search the internet, and work continuously until it solves the problem, finally emailing you the results.

No expensive API keys, no cloud data leaks, and optimized to run on consumer-grade hardware.

## ✨ Features
* **Fully Local AI:** Powered by [Ollama](https://ollama.com/), ensuring absolute privacy and zero API costs.
* **File-Triggered:** Uses `watchdog` to monitor a local `tasks.txt` file. Just save a new task to the file, and the agent wakes up.
* **Autonomous Web Research:** Equipped with DuckDuckGo search to pull real-time data from the internet.
* **ReAct Agent Loop:** Thinks, acts, and observes continuously until it determines the task is fully complete.
* **Email Notifications:** Automatically formats the final solution and delivers it straight to your inbox.

## 💻 Hardware Requirements
This project was designed to be highly accessible and runs smoothly on older or consumer-tier hardware.
* **OS:** Ubuntu / Linux (Recommended for native hardware acceleration)
* **GPU:** Minimum 8GB VRAM (e.g., NVIDIA GTX 1070) to run 7B/8B parameter models in 4-bit quantization.
* **RAM:** 16GB+ System RAM (32GB recommended).

## 🛠️ Prerequisites
1. **Python 3.10+**
2. **Ollama:** Install from [ollama.com](https://ollama.com/)
3. **An Email App Password:** If using Gmail, you must generate a 16-character [App Password](https://support.google.com/accounts/answer/185833?hl=en) to allow the script to send emails.

## 🚀 Quick Start

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/Huginn.git](https://github.com/yourusername/Huginn.git)
cd Huginn

```

**2. Install dependencies**

```bash
pip install -r requirements.txt

```

**3. Pull the recommended local model**
We recommend `qwen2.5:7b` or `llama3.1:8b` for the best balance of reasoning, tool use, and VRAM efficiency.

```bash
ollama pull qwen2.5:7b

```

**4. Set up your environment variables**
Create a `.env` file in the root directory and add your email credentials:

```env
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_16_character_app_password
EMAIL_RECEIVER=where_to_send_results@gmail.com

```

**5. Run the Agent**

```bash
python main.py

```

## 📝 Usage

Once the script is running, it will listen to the `tasks.txt` file in the project directory.

Open `tasks.txt`, write a prompt, and hit save:

> *"Research the current top 3 open-source local text-to-speech models and give me a summary of their hardware requirements."*

Check your terminal to watch the agent's thought process (Thought -> Action -> Observation). When it finishes, check your email for the final report!

## 🤝 Contributing

Contributions are welcome! If you have ideas for adding new tools (like local file reading, calculator, etc.) or optimizing the agent loop, feel free to open an issue or submit a pull request.

## 📄 License

MIT License
