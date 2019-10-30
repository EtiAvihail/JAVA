
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Main {

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

		Instances training = loadData("wind_training.txt");
		Instances testing = loadData("wind_testing.txt");

		LinearRegression linRegTraining = new LinearRegression();
		linRegTraining.buildClassifier(training);
		double trainingError = linRegTraining.calculateMSE(training);
		double testingError = linRegTraining.calculateMSE(testing);
		double curError = 0;
		System.out.println("The chosen alpha is: " + linRegTraining.m_alpha);
		System.out.println("Training error with all features is: " + trainingError);
		System.out.println("Test error with all features is: " + testingError);

		for (int i = 0; i < training.numAttributes() - 1; i++) {
			for (int j = i + 1; j < training.numAttributes() - 1; j++) {
				for (int k = j + 1; k < training.numAttributes() - 1; k++) {
					linRegTraining.curChosenThetas[0] = i;
					linRegTraining.curChosenThetas[1] = j;
					linRegTraining.curChosenThetas[2] = k;
					linRegTraining.buildClassifier(training);
					curError = linRegTraining.calculateMSEThreeThetas(training, linRegTraining.curChosenThetas);
					System.out.println(training.attribute(i).name() + "-" + training.attribute(j).name()
							+ "-" + training.attribute(k).name() + " : " + curError);
				}
			}
		}
		int[] chosenThetasArr = linRegTraining.chosenThetas;
		System.out.println("Training error the features : " + training.attribute(chosenThetasArr[0]).name() + "-"
				+ training.attribute(chosenThetasArr[1]).name() + "-" + training.attribute(chosenThetasArr[2]).name()
				+ " : " + linRegTraining.minimalErrorThreeThetas);
		linRegTraining.curChosenThetas = linRegTraining.chosenThetas;
		linRegTraining.buildClassifier(testing);
		System.out.println("Test error the features : " + training.attribute(chosenThetasArr[0]).name() + "-"
				+ training.attribute(chosenThetasArr[1]).name() + "-" + training.attribute(chosenThetasArr[2]).name()
				+ " : " + linRegTraining.calculateMSEThreeThetas(testing, linRegTraining.chosenThetas));
	}
}