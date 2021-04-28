import pickle
from skimage import io
from skimage import transform

digit_to_predict_filename = input("Input the image filename of the digit you want to predict: ")

digit_to_predict = io.imread(digit_to_predict_filename, as_gray=True)
digit_to_predict = transform.resize(digit_to_predict, (100, 100))
digit_to_predict = digit_to_predict.flatten() # Turn the digit sample bidimiensional array into a unidimensional array

print("Importing model...") 
with open("./model.pkl", "rb") as f:
    clf = pickle.load(f)

results = clf.predict([digit_to_predict,])
print(chr(results[0]))