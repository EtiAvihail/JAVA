package HomeWork3;


import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScaler {
    /**
     * Returns a scaled version (using standardized normalization) of the given data set.
     *
     * @param instances The original data set.
     * @return A scaled instances object.
     * @throws Exception
     */
    public Instances scaleData(Instances instances) throws Exception {

        Standardize standardize = new Standardize();
        standardize.setInputFormat(instances);
        return Filter.useFilter(instances, standardize);

    }
}