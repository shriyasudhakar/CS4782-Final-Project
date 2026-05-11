import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# NOTE: you must rename headers of StegExpose generated csv file to:
# file_name,above_threshold,message_size,primary_sets,chi_square,sample_pairs,rs_analysis,fusion

depths = [1,3,6]
for D in depths:
    df = pd.read_csv(f"./StegExpose/steganalysisCOCODenseD={D}.csv")
    df = df.drop(columns=['message_size', 'primary_sets', "chi_square", "sample_pairs", "rs_analysis"])
    df['truth'] = False
    df.loc[df['file_name'].str.startswith("stego_"),'truth'] = True
    # above_threshold is basically predicted
    print(df.to_string()) 


    fpr, tpr, _ = roc_curve(df['truth'], df["above_threshold"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Dense Encoder D={D} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('COCO Test Set ROC Curve with StegExpose Fusion (mean), threshold=0.2')
plt.legend()
plt.savefig("coco_dense_roc_auc.png")