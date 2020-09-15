def polarity(column):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import pandas as pd
    # apply vader
    final_results = pd.DataFrame()
    for i in range(len(column)):
        text = column[i]
    
        vs = SentimentIntensityAnalyzer().polarity_scores(text)
#        print(vs)
        results = pd.DataFrame([vs['neg'],  vs['neu'],  vs['pos'],  vs['compound']]).T
        final_results = pd.concat([final_results,results])    
    final_results.columns = ['negative', 'neutral', 'positive', 'compound']
    return final_results
