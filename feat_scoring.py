# evaluate performance of FSM
import pandas as pd
import numpy as np
import argparse

# 互斥率 for each node: 這個node，在時間上，ASR-mask跟AD-mask不同的比例
# input shape: (time-step,)
def Mutex_rate(ASR_fsm, D_fsm):                                                 # calculate the rate of "mutually exclusion"
    num = (ASR_fsm != D_fsm).sum()                                              # numerator: num of different values in feat. scoring
    return num / len(D_fsm)

# 這個node，在時間上，ASR-mask跟AD-mask皆為1的比例
# input shape: (time-step,)
def rate_11(ASR_fsm, D_fsm):                                                    # calculate rate of 2 masks being 1s
    num = ((ASR_fsm == 1) * (D_fsm == 1)).sum()                                 # numerator: num of 2 masks being 1s
    return num / len(D_fsm)

# mutual info
from sklearn.metrics import mutual_info_score
# 這個node，在時間上，ASR-mask跟AD-mask的mutual info
# input shape: (time-step,)
def MutualInfo(ASR_fsm, D_fsm):                                                 # calculate Mutual info.
    return mutual_info_score(ASR_fsm, D_fsm)

def main() -> None:
    # configs:
    # model_name: will read from model_name.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model_name', type=str, default="FSM_indiv_AMLoss_5_lmFSM", help="name of the desired model, ex: FSM_indiv_AMLoss_5_lmFSM")
    args = parser.parse_args()
    
    # load csv data: for testing only
    #df_train = pd.read_csv("./saves/results/" + args.model_name + "_train.csv")
    #df_dev = pd.read_csv("./saves/results/" + args.model_name + "_dev.csv")
    df_test = pd.read_csv("./saves/results/" + args.model_name + ".csv")
    
    # 轉成list
    """
    for i in range(len(df_train)):                                                               # for training data
        df_train.loc[i, "hidden_states"] = np.array(eval(df_train.loc[i, "hidden_states"]))
        df_train.loc[i, "lm_mask"] = np.array(eval(df_train.loc[i, "lm_mask"]))
        df_train.loc[i, "dementia_mask"] = np.array(eval(df_train.loc[i, "dementia_mask"]))
        print("\r"+ str(i+1), end="")
    """
    # turn into list for testing data: only need "hidden_states", "lm_mask", and "dementia_mask"
    for i in range(len(df_test)):                                                               # for testing data
        df_test.loc[i, "hidden_states"] = np.array(eval(df_test.loc[i, "hidden_states"]))
        df_test.loc[i, "lm_mask"] = np.array(eval(df_test.loc[i, "lm_mask"]))
        df_test.loc[i, "dementia_mask"] = np.array(eval(df_test.loc[i, "dementia_mask"]))
        print("\r"+ str(i+1), end="")
    """    
    # for TEST
    lm_masks = df_test.lm_mask
    AD_masks = df_test.dementia_mask
    num_node = np.shape(lm_masks[0][0])[1]

    # 下面幾個值針對node去算
    MEX_rates = np.zeros(num_node)
    MIs = np.zeros(num_node)
    lm_node_on_rate = np.zeros(num_node)
    AD_node_on_rate = np.zeros(num_node)
    rates_11 = np.zeros(num_node)
    #lm_mask_av = np.zeros(num_node)
    #ad_mask_av = np.zeros(num_node)

    for i in range(len(lm_masks)):                                          # for each utt
        lm_mask = np.transpose(lm_masks[i][0])                              # (time-step, 768) for ith utt --> (768, time-step)
        AD_mask = np.transpose(AD_masks[i][0])                              # (time-step, 768) for ith utt --> (768, time-step)

        for j in range(num_node):                                           # for each NODE of hidden states
            MEX_rate = Mutex_rate(lm_mask[j], AD_mask[j])
            MI = MutualInfo(lm_mask[j], AD_mask[j])
            MEX_rates[j] += MEX_rate
            MIs[j] += MI

            lm_node_on_rate[j] += sum(lm_mask[j]) / len(lm_mask[j])         # 這個node on 的比例
            AD_node_on_rate[j] += sum(AD_mask[j]) / len(AD_mask[j])

            rates_11[j] += rate_11(lm_mask[j], AD_mask[j])
            # average mask
            #lm_mask_av[j] += sum(lm_mask[j]) / len(lm_mask[j])
            #ad_mask_av[j] += sum(AD_mask[j]) / len(AD_mask[j])

    # get result avg over utt
    MEX_rates = MEX_rates / (i+1)               
    MIs = MIs / (i+1)
    lm_node_on_rate = lm_node_on_rate / (i+1)
    AD_node_on_rate = AD_node_on_rate / (i+1)
    rates_11 = rates_11 / (i+1)
    #lm_mask_av = lm_mask_av / (i+1)
    #ad_mask_av = ad_mask_av / (i+1)
    pd.DataFrame([MEX_rates, MIs, lm_node_on_rate, AD_node_on_rate, rates_11]).to_csv("./saves/results/FSM_info/" + args.model_name + '.csv')  

    ####################################
    # 每條hidden state的開關率
    ####################################
    num_utt = len(lm_masks)
    print("num_utt: ", num_utt)
    # 以下針對每個utt算
    lm_on_rate = np.zeros(num_utt)
    AD_on_rate = np.zeros(num_utt)

    for i in range(num_utt):                                        # for each utt
        lm_mask = lm_masks[i][0]                                    # (time-step, hidden_size) for ith utt
        AD_mask = AD_masks[i][0]                                    # (time-step, hidden_size) for ith utt

        time_step = np.shape(lm_mask)[0]
        print("time-step = ", time_step)
        for j in range(time_step):                                  # for each time-step
            # average mask: 每條hidden state的開關率和
            lm_on_rate[i] += sum(lm_mask[j]) / len(lm_mask[j])      # 加上 這條hidden state的開關率
            AD_on_rate[i] += sum(AD_mask[j]) / len(AD_mask[j]) 
        # i-th utt的 hidden state的開關率 的平均
        lm_on_rate[i] = lm_on_rate[i] / time_step
        AD_on_rate[i] = AD_on_rate[i] / time_step        
        # rate of toggling off
    pd.DataFrame([lm_on_rate, AD_on_rate]).to_csv("./saves/results/FSM_info/" + args.model_name + '_onRate.csv')

    ####################################
    # 針對每個node去算，800個音檔中的開關率
    # -> 每個node會有800個值，後續再去排大小畫box plot
    ####################################
    MEX_rates_df = pd.DataFrame()
    MIs_df = pd.DataFrame()
    lm_node_on_rate_df = pd.DataFrame()
    AD_node_on_rate_df = pd.DataFrame()
    rates_11_df = pd.DataFrame()
    for i in range(len(lm_masks)):              # for each utt
        # reset arrays
        # for each node of this utt
        MEX_rates = np.zeros(num_node)
        MIs = np.zeros(num_node)
        lm_node_on_rate = np.zeros(num_node)
        AD_node_on_rate = np.zeros(num_node)
        rates_11 = np.zeros(num_node)

        lm_mask = np.transpose(lm_masks[i][0])  # (time-step, num_node) for ith utt --> (num_node, time-step)
        AD_mask = np.transpose(AD_masks[i][0])  # (time-step, num_node) for ith utt --> (num_node, time-step)

        for j in range(num_node):               # for each node of hidden states
            MEX_rate = Mutex_rate(lm_mask[j], AD_mask[j])
            MI = MutualInfo(lm_mask[j], AD_mask[j])
            MEX_rates[j] = MEX_rate
            MIs[j] = MI

            lm_node_on_rate[j] = sum(lm_mask[j]) / len(lm_mask[j])
            AD_node_on_rate[j] = sum(AD_mask[j]) / len(AD_mask[j])

            rates_11[j] = rate_11(lm_mask[j], AD_mask[j])
        
        # add to dataframe col
        # y for num_node nodes
        MEX_rates_df["utt_" + str(i+1) + "_MEX_rates"] = MEX_rates
        MIs_df["utt_" + str(i+1) + "_MIs"] = MIs
        lm_node_on_rate_df["utt_" + str(i+1) + "_lm_node_on_rate"] = lm_node_on_rate
        AD_node_on_rate_df["utt_" + str(i+1) + "_AD_node_on_rate"] = AD_node_on_rate
        rates_11_df["utt_" + str(i+1) + "_rates_11"] = rates_11
    
    MEX_rates_df.to_csv("./saves/results/FSM_info/" + args.model_name + '_NodeWise_MEX_rates.csv')
    MIs_df.to_csv("./saves/results/FSM_info/" + args.model_name + '_NodeWise_MIs.csv')
    lm_node_on_rate_df.to_csv("./saves/results/FSM_info/" + args.model_name + '_NodeWise_lm_node_on_rate.csv')
    AD_node_on_rate_df.to_csv("./saves/results/FSM_info/" + args.model_name + '_NodeWise_AD_node_on_rate.csv')
    rates_11_df.to_csv("./saves/results/FSM_info/" + args.model_name + '_NodeWise_rates_11.csv')  
    """
    ##################################################################################################################################
    # masked emb. 在所有維度上的加總 & pull out each time-step's embedding
    ##################################################################################################################################
    lm_masks = df_test.lm_mask                                   # ASR masks
    AD_masks = df_test.dementia_mask                             # AD masks
    hidden_states = df_test.hidden_states                        # hidden states
    
    num_time_steps = 0                                           # count total num of time-step
    lm_sum = 0                                                   # sum up lm-masked emb.
    ad_sum = 0                                                   # sum up ad-masked emb.
    num_utt = len(lm_masks)                                      # num of utt
    lm_masked_emb_df = pd.DataFrame()                            # save lm-masked emb. for all time-step
    ad_masked_emb_df = pd.DataFrame()                            # save ad-masked emb. for all time-step
    embs_df = pd.DataFrame()                                     # save unmasked emb. for all time-step
    for i in range(num_utt):                                     # for each utt
        lm_mask = lm_masks[i][0]                                 # (time-step, hidden_size) for ith utt
        AD_mask = AD_masks[i][0]                                 # (time-step, hidden_size) for ith utt
        hidden_state = hidden_states[i][0]                       # (time-step, hidden_size) for ith utt

        lm_masked_emb = lm_mask * hidden_state                   # lm-masked emb.
        ad_masked_emb = AD_mask * hidden_state                   # ad-masked emb.

        df_lm = pd.DataFrame(lm_masked_emb)                      # to DataFrame type
        lm_masked_emb_df = pd.concat([lm_masked_emb_df, df_lm], ignore_index=True)
                                                                 # add to resulted DataFrame
        df_ad = pd.DataFrame(ad_masked_emb)                      # to DataFrame type
        ad_masked_emb_df = pd.concat([ad_masked_emb_df, df_ad], ignore_index=True)
                                                                 # add to resulted DataFrame

        df_emb = pd.DataFrame(hidden_state)                      # to DataFrame type
        embs_df = pd.concat([embs_df, df_emb], ignore_index=True)# add to resulted DataFrame

        lm_sum += lm_masked_emb.sum()                            # add to sum of lm-masked emb.
        ad_sum += ad_masked_emb.sum()                            # add to sum of ad-masked emb.

        time_step = np.shape(lm_mask)[0]                         # num of time-steps for this utt
        num_time_steps += time_step                              # add to total num

    print("Total time-steps = ", num_time_steps)
    lm_mean = lm_sum / num_time_steps                            # average over all time-steps
    ad_mean = ad_sum / num_time_steps                            # average over all time-steps
    print("lm mean: ", lm_mean)
    print("ad mean: ", ad_mean)
    
    lm_masked_emb_df.to_csv("./saves/results/FSM_info/" + args.model_name + '_lm_masked_embs.csv') 
    ad_masked_emb_df.to_csv("./saves/results/FSM_info/" + args.model_name + '_ad_masked_embs.csv') 
    
    embs_df.to_csv("./saves/results/FSM_info/" + args.model_name + '_un_masked_embs.csv') 
    print("All done")
    
if __name__ == "__main__":
    main()
