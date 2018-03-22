package HomeWork1;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	
	final int NUM_OF_STEPS = 20000;
	final double errorDif = 0.003;
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes() - 1;
		m_coefficients = new double[m_truNumAttributes];
		
		Random random = new Random();
		for (int i = 0; i < m_coefficients.length; i++) {
			m_coefficients[i] = random.nextDouble();
		}
		
		
		findAlpha(trainingData);
		System.out.println("This is the best alpha " + m_alpha);
		
		double mse = calculateMSE(trainingData);
		double prevMse = mse + 1;
		int counter = 1;
		
		while(prevMse - mse > errorDif)
		{
			counter++;
			
			m_coefficients = gradientDescent(trainingData);
			
			if(counter % 100 == 0)
			{
				 prevMse = mse;
				 mse = calculateMSE(trainingData);
				 System.out.println("mse: "+mse+"\t dif: "+(prevMse-mse));
				 
				 if(prevMse - mse < 0)
				 {
					 System.out.println("HERE");
				 }
			}
			
		}
		
		
		//TODO: complete this method
		//m_coefficients = gradientDescent(trainingData);
		
	}
	
	private void findAlpha(Instances data) throws Exception {
		double curMSEForAlpha = Double.MAX_VALUE; // change name
		double curAlpha = 0; // change to maxAlpha
		double[] copy_m_coefficients = new double[m_coefficients.length];
		
		for (int i = 0; i < copy_m_coefficients.length; i++) 
		{
			copy_m_coefficients[i] = m_coefficients[i];
		}
		
		// not reaching to 0
		for (int i = -17; i < 0; i++) 
		{
			
			//curAlpha = Math.pow(3, i);
			this.m_alpha = Math.pow(3, i);; // check this two lines
			double prevError = Double.MAX_VALUE;
			double curError = prevError;
			int j;
			
			for (j = 0; j < NUM_OF_STEPS; j++) 
			{
				m_coefficients = gradientDescent(data);
				
				if(j % 100 == 0)
				{
					curError = calculateMSE(data);
					System.out.println("Alpha is: " + m_alpha + " and the error is: " + curError);
					if(curError > prevError)
					{
						/*
						if(prevError < curMSEForAlpha)
						{
							curMSEForAlpha = prevError;
							this.m_alpha = curAlpha;
						}
						*/
						
						break;
					}
					
					prevError = curError;
				}
			}
			/*
			System.err.println("The prev is: " + prevError + " the best is: " + curMSEForAlpha);
			if(j == NUM_OF_STEPS)
			{
				if (curError < curMSEForAlpha)
				{
					curMSEForAlpha = curError;
					this.m_alpha = curAlpha;
				}
			}else
			{
				if (prevError < curMSEForAlpha)
				{
					curMSEForAlpha = prevError;
					this.m_alpha = curAlpha;
				}
			}
			*/
			//System.err.println("after update:  The prev is: " + prevError + " the best is: " + curMSEForAlpha);
			
			if(prevError < curMSEForAlpha)
			{

				System.err.println("The prev is: " + prevError + " the best is: " + curMSEForAlpha);
				if(j == NUM_OF_STEPS)
				{
					curMSEForAlpha = curError;
					//this.m_alpha = curAlpha;
					curAlpha = this.m_alpha;
				}
				else
				{
					curMSEForAlpha = prevError;
					curAlpha = this.m_alpha;
					//this.m_alpha = curAlpha;
				}
				
				System.err.println("The change " + curMSEForAlpha);
				
			}
			
			for (int k = 0; k < copy_m_coefficients.length; k++) 
			{
				m_coefficients[k] = copy_m_coefficients[k];
			}
			
			this.m_alpha = curAlpha;
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
	
		double [] m_coefficients_tmp = new double[m_coefficients.length];
		double [] m_coefficients_result = new double[m_coefficients.length];
		
		for (int i = 0; i < m_coefficients.length; i++) 
		{
			m_coefficients_tmp[i] = m_coefficients[i];
			m_coefficients_result[i] = m_coefficients[i];
		}
		
		//double mse = calculateMSE(trainingData);
		//double prevMse = mse+1;
		//int counter = 1;
		
		//while(prevMse - mse > errorDif)
		//{
		//	counter++;
			
			double sumOfErr = 0;
			double instancePredict;
			double instanceY;
			
			for (int i = 0; i < trainingData.size(); i++)
			{
				instancePredict = regressionPrediction(trainingData.instance(i));
				instanceY = trainingData.instance(i).classValue();
				
				sumOfErr += instancePredict - instanceY;
			}
			
			sumOfErr = sumOfErr / (trainingData.size()) * m_alpha;
			
			m_coefficients_tmp[0] -= sumOfErr; // Can change to mse?
			
			// Change the loop to go over num of features
			for (int i = 1; i < m_coefficients_tmp.length; i++) 
			{
				sumOfErr = 0;
				instancePredict = 0;
				instanceY = 0;
				
				// maybe from 1
				for (int j = 0; j < trainingData.size(); j++)
				{
					instancePredict = m_coefficients[0]; // Can be tricky
					
					// Maybe go -1 on the m_coefficients
					for (int k = 1; k < m_coefficients.length; k++) 
					{
						instancePredict += m_coefficients[k] * trainingData.instance(j).value(k);
					}
					
					instanceY = trainingData.instance(j).classValue();

					sumOfErr += (instancePredict - instanceY) * trainingData.instance(j).value(i);
				}
				
				sumOfErr = sumOfErr / (double)trainingData.size() * m_alpha;
				
				m_coefficients_tmp[i] -= sumOfErr; 
			}
			
			for (int i = 0; i < m_coefficients_tmp.length; i++) 
			{
				m_coefficients[i] = m_coefficients_tmp[i];
			}
			
			/*
			if(counter % 100 == 0)
			{
				 prevMse = mse;
				 mse = calculateMSE(trainingData);
				 System.out.println("mse: "+mse+"\t dif: "+(prevMse-mse));
				 
				 if(prevMse-mse < 0)
				 {
					 System.out.println("HERE");
				 }
			}
			*/
				
		//}
		
		return m_coefficients;

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
		double prediction = m_coefficients[0]; // Can be tricky
		
		// Maybe go -1 on the m_coefficients
		for (int i = 0; i < m_coefficients.length - 1; i++) 
		{
			//System.out.println(instance.value(i));
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
		double instancePredict;
		double instanceY;
		
		for (int i = 0; i < data.size(); i++)
		{
			instancePredict = regressionPrediction(data.instance(i));
			instanceY = data.instance(i).classValue();
			
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
