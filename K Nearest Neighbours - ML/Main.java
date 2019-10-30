
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.core.Instances;

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

    public static Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);
        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {

        //1st Part
        //Loading data
        Instances instances = loadData("auto_price.txt");
        FeatureScaler featureScaler = new FeatureScaler();
        Instances scaledInstances = featureScaler.scaleData(instances);


        //Shuffling the data of scaled and not scaled
        instances.randomize(new Random());
        scaledInstances.randomize(new Random());

        //Creating the Knn of the scaled and not scaled sets
        Knn knnNotScaled = new Knn();
        Knn knnScaled = new Knn();

        //2nd Part

        boolean isScaledSet = true;//in order to print scaled or not scaled

        //Getting the hyper parameters cross validation error
        knnNotScaled.iterateOnParametersNGetCVError(instances, 10, Knn.DistanceCheck.Regular);
        knnScaled.iterateOnParametersNGetCVError(scaledInstances, 10, Knn.DistanceCheck.Regular);

        //Printing the cross validation and the parameters
        knnNotScaled.printResults(!isScaledSet, knnNotScaled.getMinimalCrossValidationError(), knnNotScaled.getBestK(), knnNotScaled.getBestLpDistance(), knnNotScaled.getBestWeightScheme());
        knnScaled.printResults(isScaledSet, knnScaled.getMinimalCrossValidationError(), knnScaled.getBestK(), knnScaled.getBestLpDistance(), knnScaled.getBestWeightScheme());

        //3rd Part
        double currentCrossValidationError = 0;
        double averageTimeOfASingleFoldInCrossValidation;
        int[] numOfFolds = {159, 50, 10, 5, 3};
        for (int i = 0; i < numOfFolds.length; i++) {
            for (Knn.DistanceCheck distanceCheck : Knn.DistanceCheck.values()) {
                knnScaled.crossValidationTime = 0;
                Instances[] folders = knnScaled.createArrayOfFolds(instances, numOfFolds[i]);
                currentCrossValidationError = knnScaled.crossValidationError(scaledInstances, numOfFolds[i], folders, knnScaled.getBestK(),knnScaled.getBestWeightScheme(), knnScaled.getBestLpDistance(), distanceCheck);
                averageTimeOfASingleFoldInCrossValidation = knnScaled.crossValidationTime/numOfFolds[i];
                Knn.printPart3(distanceCheck,currentCrossValidationError,averageTimeOfASingleFoldInCrossValidation, knnScaled.crossValidationTime, numOfFolds[i]);
            }
        }
    }
}