 # 1. Create an isolated environment (do this once per project)
 python -m venv ml_env
 source ml_env/bin/activate        # Mac/Linux
 ml_env\Scripts\activate           # Windows (use this line instead)

 # 2. Install the libraries used in this book
 pip install -r requirements.txt

 # 3. Generate the practice datasets
 python make_datasets.py

# 4. Check that everything works
python ch01/check_install.py

Jupyter notebooks are the standard tool for ML development. They let you write code
in small cells, run each cell individually, see visualizations inline, and mix code with
text explanations. Every major ML tutorial and course uses Jupyter notebooks.

Your First Lines of Machine Learning Code
Let us build the real thing: a spam filter. The data is the public SMS Spam Collection,
5,574 real text messages labeled spam or genuine (‘ham’). The script counts how
often each word appears in each message, learns which words go with spam, and
then judges messages it never saw during training. You do not need to understand
