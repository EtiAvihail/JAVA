package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

	private double[] m_coefficients;
	private int m_truNumAttributes;
	double m_alpha;
	public static double minimalErrorThreeThetas = Double.MAX_VALUE;
	public static int[] curChosenThetas = new int[3];
	public static int[] chosenThetas = new int[3];

	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		if(m_alpha != 0) {
			m_truNumAttributes = 3;
			gradientDescentThreeThetas(trainingData, curChosenThetas);
			double currentError = calculateMSEThreeThetas(trainingData, curChosenThetas);
			if (currentError < minimalErrorThreeThetas) {
				int iteration = 0;
				minimalErrorThreeThetas = currentError;
				while(iteration < 3) {
					chosenThetas[iteration] = curChosenThetas[iteration];
					iteration++;
				}
			}
		}else{
			m_truNumAttributes = trainingData.numAttributes() - 1;
			findAlpha(trainingData);
			m_coefficients = new double[m_truNumAttributes + 1];
			m_coefficients = gradientDescent(trainingData);
		}
	}

	/**
	 * An implementation of findAlpha
	 * returns the minimum chosen Alpha
	 * 
	 * 
	 * @param data 
	 * @throws Exception
	 */
	private void findAlpha(Instances data) throws Exception{
		double minAlpha = 0;
		double minimalError = Integer.MAX_VALUE;
		double previousError = 0.0;
		//iterating on each one of the 17 possible alpha's and looking for the alpha with the minimal error
		for(int counter1 = -17; counter1 <= 0 ; counter1++) {
			//Guess initial values
			m_coefficients = null;
			m_coefficients = new double[m_truNumAttributes + 1];
			//Calculating current alpha
			m_alpha = Math.pow(3, counter1);
			m_coefficients = gradientDescent(data);
			//Checking if the this specific alpha's error is the minimal
			previousError = calculateMSE(data);
			if(minimalError > previousError) {
				minimalError =  previousError;
				minAlpha = m_alpha;
			}
		}//end for
		// setting the minimum alpha to be the chosen alpha
		m_alpha = minAlpha;
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
		int iteration = 0;
		double curError = 0;
		double prevError = Integer.MAX_VALUE;
		double attributeOfInstanceI = 0.0;
		double realFuncVal = 0.0;
		double innerMul = 0.0;
		double avgError = 0.0;
		double[] tempThetas = new double[m_truNumAttributes + 1];
		int counter1 = 0;
		int counter2 = 0;

		while(iteration <= 20000) {
			for(counter1 = 0 ; counter1 < trainingData.numAttributes(); counter1++ ) {
				//reseting the average for the new attribute
				avgError = 0;
				for(counter2 = 0 ; counter2 < trainingData.numInstances() ; counter2++) {
					realFuncVal = trainingData.instance(counter2).value(trainingData.classIndex());
					innerMul = regressionPrediction(trainingData.instance(counter2));
					if(counter1 == 0){//for thetha0
						avgError += innerMul - realFuncVal ;
					}else {
						attributeOfInstanceI = trainingData.instance(counter2).value(counter1-1);//Same attribute as theta in instance counter2 
						avgError += (innerMul - realFuncVal)*attributeOfInstanceI;//Calculating the sum of all instances prediction
					}
				}//inner for loop
				avgError = (avgError)/(double)trainingData.numInstances();//dividing in the number of instances (for the average)
				tempThetas[counter1] -= m_alpha*(avgError);
			}//outer for loop
			if((iteration > 100) && ((iteration % 100) == 0)) {
				curError = calculateMSE(trainingData);
				if((prevError - curError) < 0.003) {
					break;//stop if the difference between the errors is very small
				}
				else {
					prevError = curError;//define the new prevError because there is a new cycle of 100 iterations
				}
			}
			for (int i = 0; i < tempThetas.length; i++) {
				m_coefficients[i] = tempThetas[i];
			}
			iteration++;
		}
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
		double prediction = m_coefficients[0];//instantiating to the theta0 (because its not part of the sigma)
		for (int counter = 0; counter < instance.numAttributes()-1;  counter++) {
			prediction += m_coefficients[counter + 1] * instance.value(counter);
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
		double prediction = 0;
		double sumSquaredError = 0;
		Instance instance = null;
		double realVal = 0;
		for (int counter = 0; counter < data.numInstances(); counter++){
			instance  = data.instance(counter);
			prediction = regressionPrediction(instance);
			realVal = instance.value(instance.numAttributes()-1);
			sumSquaredError += Math.pow((prediction - realVal), 2);
		}
		return (sumSquaredError  / (2 * data.numInstances())); 
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

	private double[] gradientDescentThreeThetas(Instances trainingData, int[] threeThetasIndices )
			throws Exception {
		int iteration = 0;
		double curError = 0;
		double prevError = Integer.MAX_VALUE;
		double attributeOfInstanceI = 0.0;
		double realFuncVal = 0.0;
		double innerMul = 0.0;
		double avgError = 0.0;
		double[] tempThetas = new double[m_truNumAttributes + 1];
		int counter1 = 0;
		int counter2 = 0;

		while(iteration <= 20000) {
			for(counter1 = 0 ; counter1 < m_truNumAttributes + 1; counter1++ ) {
				//reseting the average for the new attribute
				avgError = 0;
				for(counter2 = 0 ; counter2 < trainingData.numInstances() ; counter2++) {
					realFuncVal = trainingData.instance(counter2).value(trainingData.classIndex());
					innerMul = regressionPredictionThreeThetas(trainingData.instance(counter2), threeThetasIndices);
					if(counter1 == 0){//for thetha0
						avgError += innerMul - realFuncVal ;
					}else {
						attributeOfInstanceI = trainingData.instance(counter2).value(threeThetasIndices[counter1-1]);//Same attribute as theta in instance counter2 
						avgError += (innerMul - realFuncVal)*attributeOfInstanceI;//Calculating the sum of all instances prediction
					}
				}//inner for loop
				avgError = (avgError)/(double)trainingData.numInstances();//dividing in the number of instances (for the average)
				tempThetas[counter1] -= m_alpha*(avgError);
			}//outer for loop
			if((iteration > 100) && ((iteration % 100) == 0)) {
				curError = calculateMSEThreeThetas(trainingData, threeThetasIndices);
				if((prevError - curError) < 0.003) {
					break;//stop if the difference between the errors is very small
				}
				else {
					prevError = curError;//define the new prevError because there is a new cycle of 100 iterations
				}
			}
			for (int i = 0; i < tempThetas.length ; i++) {
				m_coefficients[i] = tempThetas[i];
			}
			iteration++;
		}
		return m_coefficients;
	}


	public double calculateMSEThreeThetas(Instances data, int[] threeThetasIndices) throws Exception {
		double prediction = 0;
		double sumSquaredError = 0;
		Instance instance = null;
		double realVal = 0;
		for (int counter = 0; counter < data.numInstances() ; counter++){
			instance  = data.instance(counter);
			prediction = regressionPredictionThreeThetas(instance, threeThetasIndices);
			realVal = instance.value(instance.numAttributes()-1);
			sumSquaredError += Math.pow((prediction - realVal), 2);
		}
		return (sumSquaredError  / (2 * data.numInstances())); 
	}

	public double regressionPredictionThreeThetas(Instance instance, int[] threeThetasIndices) throws Exception {
		double prediction = m_coefficients[0];//instantiating to the theta0 (because its not part of the sigma)
		for (int counter = 0; counter < m_truNumAttributes;  counter++) {
			prediction += m_coefficients[counter + 1] * instance.value(threeThetasIndices[counter]);
		}
		return prediction;
	}
}
