package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MainHW1 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
		
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {

		// Load training and testing data
		Instances trainingSet = loadData("wind_training.txt");
		Instances testingSet = loadData("wind_testing.txt");
		
		//find best alpha and build classifier with all attributes
		LinearRegression linearRegression = new LinearRegression();
		linearRegression.buildClassifier(trainingSet); // train it
		
		double trainingMSE = linearRegression.calculateMSE(trainingSet); // For getting the training error
		double testingMSE = linearRegression.calculateMSE(testingSet); // For getting the testing error
		
		System.out.println("Training error with all features is: " + trainingMSE);
		System.out.println("Test error with all features is: " + testingMSE);
		System.out.println("List of all combination of 3 features and the training error:");

		int attrsize = trainingSet.numAttributes() - 1;
		double minTrainError = Double.MAX_VALUE;
		 
		int[] attrArr = new int[4];
		attrArr[3] = 14; // Always keep the y value
		 
		int[] minAttrArr = new int[4];
		minAttrArr[3] = 14;
		 
		Remove remove = new Remove(); // New instance of remove
		remove.setInvertSelection(true); // Keep the given indexes and don't remove them
		Instances filteredInstances = new Instances(trainingSet);

		int i, j, k;
		
		// Find add 3 features combinations
		for (i = 0; i < attrsize-2; i++)
		{
			attrArr[0] = i; // Set first attribute
			for (j = i+1 ; j < attrsize-1; j++)
			{
				attrArr[1] = j;// Set second attribute
				for (k = j + 1; k < attrsize; k++)
				{
					attrArr[2] = k; // Set third attribute
					remove.setAttributeIndicesArray(attrArr);
					remove.setInputFormat(trainingSet);
					filteredInstances = Filter.useFilter(trainingSet, remove); // Apply filter
					
					linearRegression.buildClassifier(filteredInstances); // Train by the filtered instances
					double filteredTrainingMSE = linearRegression.calculateMSE(filteredInstances); // For getting the 3 attributes training error
					
					System.out.println(attrArr[0] + "," + attrArr[1] + "," + attrArr[2] + ": " + filteredTrainingMSE);
					
					if(filteredTrainingMSE < minTrainError)
					{
						minTrainError = filteredTrainingMSE; // Set the min error value

						// Set the min error indexes for that value
						for (int h = 0; h < 3; h++) {
							minAttrArr[h] = attrArr[h];
						}
					}		
				}
			}
		}
		
		
		System.out.println("Training error the features " + minAttrArr[0] + "," + minAttrArr[1] + "," + minAttrArr[2] + ": " + minTrainError);

		remove.setAttributeIndicesArray(minAttrArr);
		remove.setInputFormat(testingSet);                     
		filteredInstances = Filter.useFilter(testingSet, remove); // Apply filter
		
		linearRegression.buildClassifier(filteredInstances); // train it by the min filtered instances
		double testingFilteredMSE = linearRegression.calculateMSE(filteredInstances); // For getting the 3 attributes testing error
		
		System.out.println("Test error the features " + minAttrArr[0] + "," + minAttrArr[1] + "," + minAttrArr[2] + ": " + testingFilteredMSE);
		

	}

}
