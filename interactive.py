"""
File: interactive.py
Name: Chia-Chun, Hung
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
from util import interactivePrompt
from submission import extractWordFeatures

TRAINED_WEIGHTS_FILEPATH = 'weights'


def main():
	# obtain the weights that were trained
	weights = {}
	with open(TRAINED_WEIGHTS_FILEPATH, 'r')as f:
		for line in f:
			key, value = line.split()
			weights[key] = float(value)
			# watch out: when read in file by using with open(), the data type of the content is always str!
	interactivePrompt(extractWordFeatures, weights)


if __name__ == '__main__':
	main()
