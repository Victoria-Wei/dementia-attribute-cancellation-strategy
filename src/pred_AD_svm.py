# predict AD using (masked) embeddings
import pandas as pd
import numpy as np
from numpy import nan
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from functions.Functions import ID2Label

def trainSVM(x_train, y_train, x_test, y_test, df_test, title):
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    
    svm = SVC() #class_weight='balanced')
    svm.fit(x_train_std, y_train["dementia_labels"].values)
    
    pred = svm.predict(x_test_std)
    true = y_test["dementia_labels"].values
    """
    # utt-wise results
    cm = confusion_matrix(true, pred)
    
    # save results
    df = pd.read_csv("./saves/results/SVM/results.csv")                           # read in previous results
    new_row = {'model': title + " utt-wise",
               'ACC':accuracy_score(true, pred), 'BACC':balanced_accuracy_score(true, pred), 'F1':f1_score(true, pred),
               'Sens':recall_score(true, pred), 'Spec':cm[0,0]/(cm[0,0]+cm[0,1]), 'UAR': recall_score(true, pred, average='macro')}
                                                                                  # result to save
    df2 = pd.DataFrame([new_row])
    df3 = pd.concat((df, df2), axis = 0)                                          # append row
    df3.to_csv("./saves/results/SVM/results.csv", index=False)
    """
    sorted_dict = {}                                                              # sort based on spk id
    for idx, i in enumerate(df_test.index.tolist()): 
        id_part = df_test['path'][i].split('_')                                   # split file name   
        if id_part[1] == 'PAR':                                                   # predict only on participant
            if id_part[0] not in sorted_dict.keys():                              # new spk
                sorted_dict[id_part[0]] = [pred[idx]]                             # add values to this spk
            else:
                sorted_dict[id_part[0]].append(pred[idx])                         # append to existing list

    true = []                                                                     # ground truth
    pred = []                                                                     # prediction
    for spkid in sorted_dict.keys():                                              # for each spk
        true_label = ID2Label(spkid + '_PAR')                                     # get his/her label
        true.append(true_label)                                                   # add to list

        vote = sum(sorted_dict[spkid]) / len(sorted_dict[spkid])                  # average result of predictions
        if vote > 0.5:                                                            # over half of the pred is AD
            pred.append(1)                                                        # view as AD
        else:
            pred.append(0)                                                        # view as HC
    
    cm = confusion_matrix(true, pred)
    
    # save results
    df = pd.read_csv("./saves/results/SVM/results.csv")                           # read in previous results
    new_row = {'model': title + " spkid-wise",
               'ACC':accuracy_score(true, pred), 'BACC':balanced_accuracy_score(true, pred), 'F1':f1_score(true, pred),
               'Sens':recall_score(true, pred), 'Spec':cm[0,0]/(cm[0,0]+cm[0,1]), 'UAR': recall_score(true, pred, average='macro')}
                                                                                  # result to save
    df2 = pd.DataFrame([new_row])
    df3 = pd.concat((df, df2), axis = 0)                                          # append row
    df3.to_csv("./saves/results/SVM/results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model_name', type=str, default="FSM_indiv_AMLoss_5_lmFSM", help="name of the desired model, ex: FSM_indiv_AMLoss_5_lmFSM")
    parser.add_argument('-INV', '--INV', action='store_true', default=False, help="True: train w/ INV")
    parser.add_argument('-sq', '--squeeze', type=str, default="mean", help="way to squeeze hidden_states, 'mean', 'min', and 'max'")
    args = parser.parse_args()
    sqz = args.squeeze

    # load in train / test data for certain model
    df_train = pd.read_csv("./saves/results/" + args.model_name + "_train.csv")
    df_test = pd.read_csv("./saves/results/" + args.model_name + ".csv")
    
    if not args.INV:
        print("Train w/ PAR only...")
        df_train = df_train[df_train.path.str.contains("PAR")]                        # train w/ PAR only
        print("Remaining training samples: ", len(df_train))                          # show # of utt left
        df_test = df_test[df_test.path.str.contains("PAR")]                           # train w/ PAR only
        print("Remaining testing data: ", len(df_test))                               # show # of utt left

    if 'lm_mask' in df_train.columns:                                             # model w/ mask
        # turn hidden_states & masks to list
        for idx, i in enumerate(df_train.index.tolist()):                         # for training data. i: data index; idx: 順序
            df_train.loc[i, "hidden_states"] = np.array(eval(df_train.loc[i, "hidden_states"]))
            df_train.loc[i, "lm_mask"] = np.array(eval(df_train.loc[i, "lm_mask"]))
            #df_train.loc[i, "dementia_mask"] = np.array(eval(df_train.loc[i, "dementia_mask"]))
            print("\r"+ str(idx+1), end="")
        
        re = df_train.hidden_states * df_train.lm_mask                            # 得到masked hidden_states  
        
        for idx, i in enumerate(df_train.index.tolist()):
            if sqz == "mean":
                re[i] = np.mean(re[i], axis=1)                                    # 平均成(1, hidden_size)
            elif sqz == "min":
                re[i] = np.min(re[i], axis=1)                                     # 取min，成(1, hidden_size)
            elif sqz == "max":
                re[i] = np.max(re[i], axis=1)                                     # 取max，成(1, hidden_size)
            elif sqz == "median":
                re[i] = np.median(re[i], axis=1)                                  # 取median，成(1, hidden_size)

            re[i] = re[i][0]                                                      # 轉成(hidden_size)
            #print(re[i].shape)
            print("\r"+ str(idx+1), end="")
        print(" ")
        x_train = pd.DataFrame(re, columns=["masked_hidden_states"]).masked_hidden_states.tolist() # masked_hidden_states to list
        y_train = pd.DataFrame(df_train["dementia_labels"])

        # 轉成list
        for idx, i in enumerate(df_test.index.tolist()):                          # for testing data
            df_test.loc[i, "hidden_states"] = np.array(eval(df_test.loc[i, "hidden_states"]))
            df_test.loc[i, "lm_mask"] = np.array(eval(df_test.loc[i, "lm_mask"]))
            #df_test.loc[i, "dementia_mask"] = np.array(eval(df_test.loc[i, "dementia_mask"]))
            print("\r"+ str(idx+1), end="")

        re = df_test.hidden_states * df_test.lm_mask                              # 得到masked hidden_states  
        
        for idx, i in enumerate(df_test.index.tolist()):
            if sqz == "mean":
                re[i] = np.mean(re[i], axis=1)                                    # 平均成(1, hidden_size)
            elif sqz == "min":
                re[i] = np.min(re[i], axis=1)                                     # 取min，成(1, hidden_size)
            elif sqz == "max":
                re[i] = np.max(re[i], axis=1)                                     # 取max，成(1, hidden_size)
            elif sqz == "median":
                re[i] = np.median(re[i], axis=1)                                  # 取median，成(1, hidden_size)
                
            re[i] = re[i][0]                                                      # 轉成(hidden_size)
            #print(re[i].shape)
            print("\r"+ str(idx+1), end="")
        x_test = pd.DataFrame(re, columns=["masked_hidden_states"]).masked_hidden_states.tolist() # masked_hidden_states to list
        y_test = pd.DataFrame(df_test["dementia_labels"])

        trainSVM(x_train, y_train, x_test, y_test, df_test, args.model_name + "_masked_INV_" + str(args.INV) + "_" + sqz)
    else:                                                                         # train w/ un-masked emb.
        # hidden_states to list
        for idx, i in enumerate(df_train.index.tolist()):                         # for training data
            df_train.loc[i, "hidden_states"] = np.array(eval(df_train.loc[i, "hidden_states"]))
            #df_train.loc[i, "hidden_states_before"] = np.array(eval(df_train.loc[i, "hidden_states_before"]))
            print("\r"+ str(idx+1), end="")

        # hidden_states to list
        for idx, i in enumerate(df_test.index.tolist()):                              # for testing data
            df_test.loc[i, "hidden_states"] = np.array(eval(df_test.loc[i, "hidden_states"]))
            #df_test.loc[i, "hidden_states_before"] = np.array(eval(df_test.loc[i, "hidden_states_before"]))
            print("\r"+ str(idx+1), end="")
    
        # for training data
        re = df_train.hidden_states                                                   # 直接用 hidden_states  

        for idx, i in enumerate(df_train.index.tolist()):  
            #print(re[i].shape)
            if sqz == "mean":
                re[i] = np.mean(re[i], axis=1)                                        # 平均成(1, hidden_size)
            elif sqz == "min":
                re[i] = np.min(re[i], axis=1)                                         # 取min，成(1, hidden_size)
            elif sqz == "max":
                re[i] = np.max(re[i], axis=1)                                         # 取max，成(1, hidden_size)
            elif sqz == "median":
                re[i] = np.median(re[i], axis=1)                                      # 取median，成(1, hidden_size)

            re[i] = re[i][0]                                                          # 轉成(hidden_size)
            #print(re[i].shape)
            print("\r"+ str(idx+1), end="")
        print(" ")

        x_train = pd.DataFrame(re, columns=["hidden_states"]).hidden_states.tolist()  # hidden_states to list
        y_train = pd.DataFrame(df_train["dementia_labels"])
        
        # for testing data
        re = df_test.hidden_states                                                    # 直接用 hidden_states  

        for idx, i in enumerate(df_test.index.tolist()):  
            if sqz == "mean":
                re[i] = np.mean(re[i], axis=1)                                        # 平均成(1, hidden_size)
            elif sqz == "min":
                re[i] = np.min(re[i], axis=1)                                         # 取min，成(1, hidden_size)
            elif sqz == "max":
                re[i] = np.max(re[i], axis=1)                                         # 取max，成(1, hidden_size)
            elif sqz == "median":
                re[i] = np.median(re[i], axis=1)                                      # 取median，成(1, hidden_size)
                
            re[i] = re[i][0]                                                          # 轉成(hidden_size)
            #print(re[i].shape)
            print("\r"+ str(idx+1), end="")
        print(" ")

        x_test = pd.DataFrame(re, columns=["hidden_states"]).hidden_states.tolist()   # hidden_states to list
        y_test = pd.DataFrame(df_test["dementia_labels"])
        
        trainSVM(x_train, y_train, x_test, y_test, df_test, args.model_name + "_ori_INV_" + str(args.INV) + "_" + sqz)


if __name__ == "__main__":
    main()
    print("All done.")
