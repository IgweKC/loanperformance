## Loan performance prediction
### predicts future loan performance. Disclaimer: This is a mini-project end-to-end project and should be optimized before production.

#### Usage
- _First ensure you have python > 3.10. Please install the requiremnts.txt file (e.g., pip install -r requirements.txt). advisable to create a virtual environment first_
- _to test the program, run the setup.py. Alternatively, run python src/main.py from the cmd_

#### Objective
- A model that can be used to predict future loan performance and set interest rate 

#### Key Assumption 
- The model is for previous borrowers (not a new borrower). This assumption is fundamental to the data we can use to train the model.
- Other assumptions are sensitive to the business and will be communicated via email as this is in a public space

#### The following was covered
- EDA: see the notbook folder
- Model
- Additional Statistics
      -   Used Kolmogorov Smirnov statistics to make sense of the prediction distribution  and placed decision-bound for Default vs Not.
- Evaluation

#### Important folder names and level
- **artefacts**: automatically generated. holds your artefacts (files, model.pkl). Top-level
- **log**: automatically generated for logging as usual.NOTE this is in the .ignore file and can't show on github
- **evaluate**: automatically generated: results produced during your run
- **notebook**: contains two notebooks that show the experiment and options explored. Top level

_have fun_

