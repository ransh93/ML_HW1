package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	
	final int NUM_OF_STEPS = 20000;
	final double ERROR_DIFF = 0.003;
	final int CHECK_POINT = 100;
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha = 0;
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes() - 1;
		m_coefficients = new double[m_truNumAttributes + 1];
		
		// Reset coefficients array
		setInitialCoefficients(1);

		// If the given alpha is 0, we need to find an alpha
		if(m_alpha == 0)
		{
			findAlpha(trainingData);
			System.out.println("The chosen alpha is: " + m_alpha);
		}
		
		// Reset coefficients array - can be changed to another point and the process will work correctly
		setInitialCoefficients(1);
		
		// Run gradient descent progress
		gradientDescentProgress(trainingData);
		
	}
	
	
	private void gradientDescentProgress(Instances trainingData) throws Exception
	{
		double mse = calculateMSE(trainingData);
		
		//ensure that the while loop will be executed at the first call - and than prevMse will update correctly
		double prevMse = mse + ERROR_DIFF + 1; 
		int counter = 1;
		
		// run until the change in the error from the last iteration of gradientDescent is smaller then errorDif
		while(prevMse - mse > ERROR_DIFF)
		{
			counter++;
			
			m_coefficients = gradientDescent(trainingData);
			
			if(counter % CHECK_POINT == 0)
			{
				 prevMse = mse;
				 mse = calculateMSE(trainingData);
			}
			
		}
	}
	
	//set the coefficients by one same value for all of them
	private void setInitialCoefficients(double val)
	{
		for (int i = 0; i < m_coefficients.length; i++) 
		{
			m_coefficients[i] = val;
		}	
	}
	
	private void findAlpha(Instances data) throws Exception {
		double minErrorValue = Double.MAX_VALUE;
		double alphaValForMinError = 0;
		double[] copy_m_coefficients = new double[m_coefficients.length];
		
		// copy the coefficients that each alpha should start with because the global array - m_coefficients
		// will be changed during the process over every chosen alpha 
		for (int i = 0; i < copy_m_coefficients.length; i++) 
		{
			copy_m_coefficients[i] = m_coefficients[i];
		}
		
		// loop over the all optional alpha values that we want to check
		for (int i = -17; i < 0; i++) 
		{
			int stepCounter;
			double prevError = Double.MAX_VALUE;
			double curError = prevError;
			
			m_alpha = Math.pow(3, i);
			
			for (stepCounter = 0; stepCounter < NUM_OF_STEPS; stepCounter++) 
			{
				m_coefficients = gradientDescent(data);
				
				if(stepCounter % CHECK_POINT == 0)
				{
					curError = calculateMSE(data);
					
					// check if the error start to grow when doing gradientDescent
					if(curError > prevError)	
						break;
										
					prevError = curError;
				}
			}
			
			// update alphaValForMinError and minErrorValue if we found better alpha than we saw until now
			if(prevError < minErrorValue)
			{
				// if we did NUM_OF_STEPS times gradientDecsent the curError is the minimal error value that we found
				// and if we did less than that - than curError > prevError so prevError is the minimal error value that we found
				if(stepCounter == NUM_OF_STEPS)
					minErrorValue = curError;
				else
					minErrorValue = prevError;
				
				alphaValForMinError = m_alpha;
			}
			
			// update back m_coefficients to the starting point of each alpha
			for (int k = 0; k < copy_m_coefficients.length; k++) 
			{
				m_coefficients[k] = copy_m_coefficients[k];
			}
			
			m_alpha = alphaValForMinError;
		}
	}
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData)
			throws Exception {
	
		double partialDerivative = 0;
		double gradientDescentValue = 0;
		double [] m_coefficients_tmp = new double[m_coefficients.length];
		
		for (int i = 0; i < m_coefficients_tmp.length; i++) 
		{
			partialDerivative = calculatePartialDerivative(trainingData, i);
			gradientDescentValue = m_alpha * partialDerivative;
			
			// Update teta vector for the right attribute
			m_coefficients_tmp[i] = m_coefficients[i] - gradientDescentValue; 

		}

		// update the all coefficients in the end of one gradient descent 
		m_coefficients = m_coefficients_tmp;
		
		return m_coefficients;

	}
	/**
	 * Gets training data and atribute index and calculate the directional derivative value when the direction is
	 * the given attribute 
     *
	 * @return directional derivative value
	 * @throws Exception
	 */
	private double calculatePartialDerivative(Instances trainingData, int attributeIndex) throws Exception {
		double sumOfSinglePartialDerv = 0;
		double instancePredict = 0;
		double instanceY = 0;
		double singlePartialDerivative = 0;
		
		for (int i = 0; i < trainingData.size(); i++)
		{
			instancePredict = regressionPrediction(trainingData.instance(i));
			instanceY = trainingData.instance(i).value(m_ClassIndex);
			
			singlePartialDerivative = instancePredict - instanceY;
			
			// The partial derivative of the first attribute is 1
			if(attributeIndex != 0)
				singlePartialDerivative *= trainingData.instance(i).value(attributeIndex - 1);
			
			sumOfSinglePartialDerv += singlePartialDerivative;
		}
		
		return sumOfSinglePartialDerv / trainingData.size();
	}
	
	
	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		double prediction = m_coefficients[0];
		
		for (int i = 0; i < m_coefficients.length - 1; i++) 
		{
			prediction += m_coefficients[i+1] * instance.value(i);
		}
		
		return prediction;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {

		double sumOfErr = 0;
		double instancePredict = 0;
		double instanceY = 0;
		
		for (int i = 0; i < data.size(); i++)
		{
			instancePredict = regressionPrediction(data.instance(i));
			instanceY = data.instance(i).value(m_ClassIndex);
			
			sumOfErr += Math.pow(instancePredict - instanceY, 2);
		}
		
		return sumOfErr/(2*data.size());
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
