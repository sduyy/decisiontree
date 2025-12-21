import pandas as pd
input_txt = "decisiontree/predict.txt"
with open(input_txt, "r") as txt_file:
    lines = txt_file.readlines()

data = {
    "ID": range(1, len(lines) + 1),
    "Label": [line.strip() for line in lines],
}
submission_df = pd.DataFrame(data)
submission_df.to_csv("decisiontree/submission.csv", index=False)
