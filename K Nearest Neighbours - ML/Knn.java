package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Collections;
import java.util.PriorityQueue;

class DistanceCalculator {

    public double limit;



    /**
     * Constructor for DistanceCalculator class that initializes limit parameter (current Threshold of the farest neighbour)
     */
    public DistanceCalculator() {
        this.limit = Double.MAX_VALUE;
    }
    /**
     * We leave it up to you whether you want the distance method to get all relevant
     * parameters(lp, efficient, etc..) or have it has a class variables.
     * @param distanceCheck current distance method
     * @param distType current distance type
     * @param one current instance in order to calculate the distance
     * @param two current instance in order to calculate the distance
     */
    public double distance(Instance one, Instance two, Knn.LpDistanceType distType, Knn.DistanceCheck distanceCheck) {
        switch (distType) {

            case One:
                switch (distanceCheck) {
                    case Regular:
                        return lpDistance(one, two, 1);
                    case Efficient:
                        return efficientLpDistance(one, two, 1, this.limit);
                    default:
                        return lpDistance(one, two, 1);
                }

            case Two:
                switch (distanceCheck) {
                    case Regular:
                        return lpDistance(one, two, 2);
                    case Efficient:
                        return efficientLpDistance(one, two, 2, this.limit);
                    default:
                        return lpDistance(one, two, 2);
                }

            case Three:
                switch (distanceCheck) {
                    case Regular:
                        return lpDistance(one, two, 3);
                    case Efficient:
                        return efficientLpDistance(one, two, 3, this.limit);
                    default:
                        return lpDistance(one, two, 3);
                }

            case Infinity:

                switch (distanceCheck) {
                    case Regular:
                        return lInfinityDistance(one, two);
                    case Efficient:
                        return efficientLInfinityDistance(one, two, this.limit);
                    default:
                        return lInfinityDistance(one, two);
                }

            default:
                return lpDistance(one, two, 1);
        }
    }

    /**
     * Returns the Lp distance between 2 instances.
     *
     * @param one current instance in order to calculate the distance
     * @param two current instance in order to calculate the distance
     * @param p the current distance method
     * @return  the distance in Lp Regular method
     */
    private static double lpDistance(Instance one, Instance two, int p) {
        double sum = 0.0;
        double root = 1 / (double) p;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            if (i != one.classIndex()) {
                sum += Math.abs(Math.pow((one.value(i) - two.value(i)), p));
            }
        }
        return Math.pow(sum, root);
    }

    /**
     * Returns the L infinity distance between 2 instances.
     *
     * @param one current instance in order to calculate the distance
     * @param two current instance in order to calculate the distance
     * @return the distance in infinity Regular method
     */
    private static double lInfinityDistance(Instance one, Instance two) {
        double maxDistance = 0.0;
        double curDistance;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            if (i != one.classIndex()) {
                curDistance = Math.abs(one.value(i) - two.value(i));
                maxDistance = (curDistance > maxDistance) ? curDistance : maxDistance;
            }
        }
        //if its above the kth distance threshold
        return maxDistance;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one current instance in order to calculate the distance
     * @param two current instance in order to calculate the distance
     * @param p the current p
     * @return the distance in Lp efficient method
     */
    private static double efficientLpDistance(Instance one, Instance two, int p, double maxDistance) {
        double difference;
        double sum = 0;
        double root = 1 / (double) p;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            difference = Math.pow((Math.abs(one.value(i) - two.value(i))), p);
            sum += difference;
            //if its above the kth distance threshold
            if ((sum > Math.pow(maxDistance, p))) {
                return Double.MAX_VALUE;
            }
        }
        return Math.pow(sum, root);
    }


    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one current instance in order to calculate the distance
     * @param two current instance in order to calculate the distance
     * @param maxDistanceOfNeighbor the current threshold
     * @return the distance in infinity efficient method
     */
    private static double efficientLInfinityDistance(Instance one, Instance two, double maxDistanceOfNeighbor) {
        double maxDistance = 0.0;
        double curDistance;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            curDistance = Math.abs(one.value(i) - two.value(i));
            maxDistance = (curDistance > maxDistance) ? curDistance : maxDistance;
            if ((maxDistance > maxDistanceOfNeighbor)) {
                return Double.MAX_VALUE;
            }
        }
        //if its above the kth distance threshold
        return maxDistance;
    }
}

public class Knn implements Classifier {


    class InstanceWithDistance implements Comparable<InstanceWithDistance> {
        int instanceIndex;
        double distance;


        /**
         * Constructs the instance with the given index and distance
         *
         * @param index
         * @param distance
         * @return
         */

        public InstanceWithDistance(int index, double distance) {
            this.instanceIndex = index;
            this.distance = distance;
        }

        /**
         * Constructs the instance with the given index and distance
         *
         * @param instance
         * @return the indicator which says if its bigger or smaller then the current instance.
         */
        @Override
        public int compareTo(InstanceWithDistance instance) {
            if (this.distance < instance.distance) {
                return -1;
            } else {
                return 1;
            }
        }
    }

    //    public enum DistanceCheck {Efficient, Regular}
    public enum DistanceCheck {
        Efficient, Regular
    }

    public enum WeightScheme {Weighted, Not_Weighted}

    public enum LpDistanceType {One, Two, Three, Infinity}

    double crossValidationTime;
    private int bestK;
    private double minimalCrossValidationError = Double.MAX_VALUE;
    private LpDistanceType bestLpDistance;
    private WeightScheme bestWeightScheme;
    public Instances m_trainingInstances;
    int numOfInstancesInFold;
    public static int printCounter = 0;

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
    }
    /**
     * returns the minimal cross validation error
     * @returns the minimal cross validation error
     */
    public double getMinimalCrossValidationError() {
        return minimalCrossValidationError;
    }
    /**
     * returns the best K
     * @returns best K
     */
    public int getBestK() {
        return bestK;
    }

    /**
     * returns the best Lp Distance
     * @returns the best Lp Distance
     */
    public LpDistanceType getBestLpDistance() {
        return bestLpDistance;
    }

    /**
     * returns the best weightScheme
     * @returns the best weightScheme
     */
    public WeightScheme getBestWeightScheme() {
        return bestWeightScheme;
    }

    /**
     * Iterating over the Hyper parameters and calculating the best hyper parameters via the cross validation error
     * @param instances the instances set
     * @param num_of_folds  number of folds
     * @param distanceCheck weight scheme
     * @throws exception
     */
    public void iterateOnParametersNGetCVError(Instances instances, int num_of_folds, DistanceCheck distanceCheck) throws Exception {
        Instances[] folders = createArrayOfFolds(instances, num_of_folds);
        double curCrossValidationError;
        //Iterating on the values of k 1<=i<=20
        for (int k = 1; k <= 20; k++) {
            //Iterating on the lpDistances
            for (LpDistanceType lpDistanceType : LpDistanceType.values()) {
                //Iterating on the weight methods
                for (WeightScheme weightingMethod : WeightScheme.values()) {
                    curCrossValidationError = crossValidationError(instances, num_of_folds, folders, k, weightingMethod, lpDistanceType, distanceCheck);
                    isBestParameters(curCrossValidationError, k, lpDistanceType, weightingMethod);
                }
            }
        }
    }

    /**
     * Printing the results of the hyper parameters, scaled and not scaled
     * @param isScaledSet Boolean in order to determine if its scaled or not scaled
     * @param minimalCrossValidationError giving the minimal cross validation error
     * @param bestK giving the best k
     * @param bestLpDistance giving the best Lp distance
     * @param bestWeightScheme giving the best weight scheme
     */
    public void printResults(boolean isScaledSet, double minimalCrossValidationError, int bestK, LpDistanceType bestLpDistance, WeightScheme bestWeightScheme) {

        if (!isScaledSet) {

            System.out.println("----------------------------");
            System.out.println("Results for original dataset:");
            System.out.println("----------------------------");
            System.out.println("Cross validation error with K = " + bestK + ", lp = " + bestLpDistance + " , majority function = \n" + bestWeightScheme + " for auto_price data is: " + minimalCrossValidationError);

        } else {

            System.out.println("----------------------------");
            System.out.println("Results for scaled dataset:");
            System.out.println("----------------------------");
            System.out.println("Cross validation error with K = " + bestK + ", lp = " + bestLpDistance + " , majority function = \n" + bestWeightScheme + " for auto_price data is: " + minimalCrossValidationError);
        }
    }


    /**
     * Printing the results of part 3.
     * @param distanceCheck giving the current distance scheme
     * @param currentCrossValidationError giving current cross validation error
     * @param averageTimeOfASingleFoldInCrossValidation giving the average time for this current fold
     * @param crossValidationTime giving the sum of the cross validation error calculations of all the current folds together.
     * @param num_of_folds giving the current number of folds
     */
    public static void printPart3(DistanceCheck distanceCheck, double currentCrossValidationError, double averageTimeOfASingleFoldInCrossValidation, double crossValidationTime, double num_of_folds) {
        if (printCounter % 2 == 0) {
            System.out.println("----------------------------");
            System.out.println("Results for " + num_of_folds + " dataset:");
            System.out.println("----------------------------");
        }
        System.out.println("Cross validation error of " + distanceCheck + " knn on auto_price dataset is " + currentCrossValidationError + "\n" +
                "and the average elapsed time is " + averageTimeOfASingleFoldInCrossValidation + "\n" +

                "The total elapsed time is: " + crossValidationTime + "\n");
        printCounter++;
    }


    /**
     * returning the current training set (all the instances that we don't observe right now), that we as potential neighbours.
     *
     * @param instances the given set of instances
     * @param folders an array of folders of instances
     * @param curFold current observed fold
     * @param num_of_fold current number of folds
     * @return a set of the training set of the instances
     */
    private Instances getTraining(Instances instances, Instances[] folders, int curFold, int num_of_fold) {
        Instances training = new Instances(instances, numOfInstancesInFold);
        //adding all the instances to the 90% training set that are not the current fold
        for (int i = 0; i < num_of_fold; i++) {
            if (i != (curFold)) {
                training.addAll(folders[i]);
            }
        }
        return training;
    }


    /**
     * Calculates the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all instances.
     *
     * @param instances the given set of instances
     * @param folders an array of folders of instances
     * @param foldNum the current number of observed fold
     * @param k current number of folds
     * @param weightingMethod the current weighting scheme
     * @param lpDistanceType the current lp distance
     * @param distanceCheck the current distance method
     * @return the current cross validation error
     */
    public double calcAvgError(Instances instances, Instances[] folders, int foldNum, int num_of_folds, int k, WeightScheme weightingMethod, LpDistanceType lpDistanceType, DistanceCheck distanceCheck) {
        m_trainingInstances = getTraining(instances, folders, foldNum, num_of_folds);
        Instance[] curNearestNeighbours;
        double curWeightedAverage;
        double curError;
        double curAverage;
        double curInstanceClassValue;
        Instance curInstance;
        double errorsSumOfCurrentFold = 0;//zeroing the sum
        //Iterating on each of the instances in the current fold
        for (int i = 0; i < folders[foldNum].numInstances(); i++) {
            curInstance = folders[foldNum].instance(i);

            //the case of weighted average
            if (weightingMethod.equals(WeightScheme.Weighted)) {
                curNearestNeighbours = findNearestNeighbors(curInstance, k, lpDistanceType, distanceCheck);
                curWeightedAverage = getWeightedAverageValue(curInstance, curNearestNeighbours, lpDistanceType, distanceCheck);
                curInstanceClassValue = curInstance.classValue();
                curError = Math.abs(curInstanceClassValue - curWeightedAverage);

            } else {

                //the case of not weighted average
                curNearestNeighbours = findNearestNeighbors(curInstance, k, lpDistanceType, DistanceCheck.Regular);
                curAverage = getAverageValue(curInstance, curNearestNeighbours);
                curInstanceClassValue = curInstance.classValue();
                curError = Math.abs(curInstanceClassValue - curAverage);
            }
            //summing the errors on the current fold in order to evaluate the fold's average error
            errorsSumOfCurrentFold += curError;
        }

        return errorsSumOfCurrentFold / folders[foldNum].size();
    }

    /**
     * Creates an array of instances which divided into folder of instances
     *
     * @param instances the given set of instances
     * @param num_of_folds the current number of folds
     * @return array of instances that each cell in an folder of instances
     */
    public Instances[] createArrayOfFolds(Instances instances, int num_of_folds) {

        Instances[] folders = new Instances[num_of_folds];
        //calculating the size of the folders in the folder array
        double size = ((double) instances.numInstances() / num_of_folds) + (double) instances.numInstances() % num_of_folds;
        //calculating the number of instances in each one of the folders
        numOfInstancesInFold = (int) Math.round((double) instances.numInstances() / num_of_folds);
        //Assigning the values into the folder array, empty values
        for (int i = 0; i < folders.length; i++) {
            folders[i] = new Instances(instances, (int) size);
        }
        //adding the instances to each one of the folds
        int counterToAddInstancesAtLastFold = 0;
        Instance curInstance;
        for (int n = 0; n < num_of_folds - 1; n++) {
            for (int m = 0; m < numOfInstancesInFold; m++) {
                curInstance = instances.instance(counterToAddInstancesAtLastFold);
                folders[n].add(curInstance);
                counterToAddInstancesAtLastFold = counterToAddInstancesAtLastFold + 1;
            }
        }
        //for the last fold, adding the remainder of the instances
        for (int n = counterToAddInstancesAtLastFold; n < instances.numInstances(); n++) {
            curInstance = instances.instance(n);
            folders[folders.length - 1].add(curInstance);
        }
        return folders;

    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     *
     * @param instances     Instances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @param k current number of folds
     * @param weightingMethod the current weighting scheme
     * @param lpDistanceType the current lp distance
     * @param distanceCheck the current distance method
     * @return The cross validation error.
     * @throws exception
     */

    public double crossValidationError(Instances instances, int num_of_folds, Instances[] folders, int k, WeightScheme weightingMethod, LpDistanceType lpDistanceType, DistanceCheck distanceCheck) throws Exception {
        double curCrossValidationError = 0;
        double startingTimeOfValidation;
        //folding the given data into num_of_folds folds
        for (int foldNum = 0; foldNum < num_of_folds; foldNum++) {
            startingTimeOfValidation = System.nanoTime();
            curCrossValidationError += calcAvgError(instances, folders, foldNum, num_of_folds, k, weightingMethod, lpDistanceType, distanceCheck);//summing up all folds avgs in order to calculate the average
            crossValidationTime += (System.nanoTime() - startingTimeOfValidation);
        }
        return curCrossValidationError / num_of_folds;//returning the Average
    }

    /**
     * Testing if the given cross validation error is the best, if yes, it assigns its parameters to be the best parameters
     * @param curCrossValidationError current cross validation error
     * @param k current number of folds
     * @param weightingMethod the current weighting scheme
     * @param lpDistanceType the current lp distance
     */
    public void isBestParameters(double curCrossValidationError, int k, LpDistanceType lpDistanceType, WeightScheme weightingMethod) {
        if (curCrossValidationError < minimalCrossValidationError) {
            minimalCrossValidationError = curCrossValidationError;
            bestK = k;
            bestLpDistance = lpDistanceType;
            bestWeightScheme = weightingMethod;
        }
    }

    /**
     * Finds the k nearest neighbors.
     * @param instance
     * @param lpDistanceType the current lp distance
     * @param distanceCheck the current distance method
     */

    public Instance[] findNearestNeighbors(Instance instance, int k, LpDistanceType p, DistanceCheck distanceCheck) {
        PriorityQueue<InstanceWithDistance> queue = new PriorityQueue<InstanceWithDistance>(Collections.reverseOrder());
        double distance;
        Instance[] knn = new Instance[k];
        InstanceWithDistance instWithDistFarest = null;
        DistanceCalculator dist = new DistanceCalculator();
        for (int i = 0; i < m_trainingInstances.numInstances(); i++) {
            distance = dist.distance(m_trainingInstances.instance(i), instance, p, distanceCheck);
            if (i < k) {
                queue.add(new InstanceWithDistance(i, distance));
                instWithDistFarest = queue.peek();
            } else if (distance < instWithDistFarest.distance) {
                queue.remove(instWithDistFarest);
                queue.add(new InstanceWithDistance(i, distance));
                instWithDistFarest = queue.peek();
                dist.limit = instWithDistFarest.distance;
            }
        }
        int index;
        InstanceWithDistance curInstance;
        for (int i = k; i > 0; i--) {
            curInstance = queue.poll();
            index = curInstance.instanceIndex;
            knn[i - 1] = m_trainingInstances.instance(index);
        }
        return knn;
}

    /**
     * Calculates the average value of the given elements in the collection.
     *
     * @param instance given instance in order to calculate the average value
     * @param knn given array of instances
     * @return the average value of the given elements in the array.
     */
    public double getAverageValue(Instance instance, Instance[] knn) {
        int k = knn.length;
        double sigma = 0.0;
        for (int i = 0; i < k; i++) {
            sigma += knn[i].classValue();
        }
        return sigma / k;
    }


    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @param newDataPoint a given instance in order to calculate the average value
     * @param knn given array of instances
     * @param p given distance type
     * @param distanceCheck given distance method
     * @return the weighted average of the target values of all the elements in the array with respect to their distance from a specific instance.
     */
    public double getWeightedAverageValue(Instance newDataPoint, Instance[] knn, LpDistanceType p, DistanceCheck
            distanceCheck) {

        double sigma = 0.0;
        int k = knn.length;
        double wi;
        double sumOfWeights = 0.0;
        double distance;
        DistanceCalculator dist = new DistanceCalculator();
        for (int i = 0; i < k; i++) {

            distance = dist.distance(knn[i], newDataPoint, p, distanceCheck);
            //case neighbor with distance zero then your prediction = neighbor target value
            if (distance == 0) {
                return knn[i].classValue();
            }

            wi = 1 / Math.pow(distance, 2);
            sumOfWeights += wi;
            sigma += (wi * knn[i].classValue());

        }
        return sigma / sumOfWeights;
    }


    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override

    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}