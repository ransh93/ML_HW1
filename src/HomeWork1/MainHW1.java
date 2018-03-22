package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

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
		

		
		
		//load data
		Instances trainingSet = loadData("C:\\Users\\Ransh\\Desktop\\Study\\ML\\HW1\\HW1\\wind_training.txt");
		Instances testingSet = loadData("C:\\Users\\Ransh\\Desktop\\Study\\ML\\HW1\\HW1\\wind_testing.txt");
		
		for (int i = 0; i < 3; i++) {
			System.out.println(trainingSet.instance(i));
		}
		System.out.println("------------------");
		
		System.out.println("------------------");
		Instance myFirstInstance = trainingSet.instance(0);
		System.out.println(myFirstInstance.classValue());
		System.out.println("------------------");
		
		for (int i = 0 ; i < trainingSet.numAttributes(); i++) {
			Attribute attribute = trainingSet.attribute(i);
			System.out.println(attribute);
		}
		
		//find best alpha and build classifier with all attributes
		LinearRegression linearRegression = new LinearRegression();
		linearRegression.buildClassifier(trainingSet); // train it
		double trainingMSE = linearRegression.calculateMSE(trainingSet); // For getting the training error
		double testingMSE = linearRegression.calculateMSE(testingSet); // For getting the testing error
		
		System.out.println("trainingMSE: " + trainingMSE);
		System.out.println("testingMSE: " + testingMSE);
		

		
   		//build classifiers with all 3 attributes combinations
		
	}

}
