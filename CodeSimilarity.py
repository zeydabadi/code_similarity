#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:41:44 2018

@author: Mahmoud Zeydabadinezhad

This python code implements uses three string-based methods and one knowledge-based method to calculate the similarity between three pyhton codes.
The python codes should be downaloded from the tinyurl.com/BMI500
"""
import numpy as np
import fnmatch as fn
import Levenshtein as L
import sys
import os
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)


def CodeSimilarity(DataPath):
# Listing the python codes.
    PythonCodePath = []
    FileName = []
#DataPath = '/home/mahmoud/deid2/data'
    Students = os.listdir(DataPath) 
    print("Some of the codes for removing age are:")
    for name in Students:
        Files = os.listdir(DataPath+'/'+name+'/'+'python')
        for file in Files:
            if fn.fnmatch(file, '*ge*.py'):
                PythonCodePath.append(DataPath+'/'+name+'/'+'python'+'/'+file)
                FileName.append(file)
                print(file)
    
    
    # Reading the python codes into a list.
    PythonCode = []
    for Code in PythonCodePath:
        with open(Code) as f:
            PythonCode.append(f.read())
            
    # Finding the similarity between codes by using sentence embedding
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      PythonCode_embeddings = session.run(embed(PythonCode))
    
      for i, PythonCode_embedding in enumerate(np.array(PythonCode_embeddings).tolist()):
        print("Embedding size: {}".format(len(PythonCode_embedding)))
        PythonCode_embedding_snippet = ", ".join(
            (str(x) for x in PythonCode_embedding[:3]))
        print("Embedding: [{}, ...]\n".format(PythonCode_embedding_snippet))
        
    def plot_similarity(labels, features, rotation):
      corr = np.inner(features, features)
      sns.set(font_scale=1.2)
      #print(corr)
      g = sns.heatmap(
          corr,
          xticklabels=labels,
          yticklabels=labels,
          vmin=0,
          vmax=1,
          cmap="YlOrRd")
      g.set_xticklabels(labels, rotation=rotation)
      g.set_title("Semantic Textual Similarity")
    
    
    def run_and_plot(session_, input_tensor_, PythonCode_, encoding_tensor):
      code_embeddings_ = session_.run(
          encoding_tensor, feed_dict={input_tensor_: PythonCode_})
      plot_similarity(FileName, code_embeddings_, 90)
    
    similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
    similarity_code_encodings = embed(similarity_input_placeholder)
    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      session.run(tf.tables_initializer())
      run_and_plot(session, similarity_input_placeholder, PythonCode,
                   similarity_code_encodings)
      
    # Finding the similarity between codes by using Jaro Similarity, Levenshtein Distance, and Ratio.
    JaroMat  = np.zeros((len(PythonCode),len(PythonCode)), dtype=np.float32)
    LevenMat = np.zeros((len(PythonCode),len(PythonCode)), dtype=np.float32)
    SimilarityMat = np.zeros((len(PythonCode),len(PythonCode)), dtype=np.float32)
    for m in range(0,len(PythonCode)):
        for n in range(0,len(PythonCode)):
            JaroMat[m][n]  = L.jaro(PythonCode[m],PythonCode[n]) # Jaro string similarity metric
            LevenMat[m][n] = L.distance(PythonCode[m],PythonCode[n]) # Absolute Levenshtein distance
            SimilarityMat[m][n] = L.ratio(PythonCode[m],PythonCode[n]) # Similarity of two strings
    
    #Plotting Jaro string Similarity   
    sns.set(font_scale=1.2)
    g = sns.heatmap(
          JaroMat,
          xticklabels=FileName,
          yticklabels=FileName,
          vmin=0,
          vmax=1,
          cmap="YlOrRd")
    g.set_xticklabels(FileName, rotation=90)
    g.set_title("Jaro string Similarity")
    
    #Plotting Absolute Levenshtein distance
    sns.set(font_scale=1.2)
    g = sns.heatmap(
          LevenMat,
          xticklabels=FileName,
          yticklabels=FileName,
          cmap="YlOrRd")
    g.set_xticklabels(FileName, rotation=90)
    g.set_title("Absolute Levenshtein distance")
    
    # Plotting Similarity of two strings
    sns.set(font_scale=1.2)
    g = sns.heatmap(
          SimilarityMat,
          xticklabels=FileName,
          yticklabels=FileName,
          vmin=0,
          vmax=1,
          cmap="YlOrRd")
    g.set_xticklabels(FileName, rotation=90)
    g.set_title("Similarity of two strings")

if __name__== "__main__":
    if len(sys.argv) == 2:
        CodeSimilarity(sys.argv[1])
    else:
        sys.exit("\nUsage: CodeSimilarity \"path to unzipped BMI500 deid python codes\" \n\n\n\n")