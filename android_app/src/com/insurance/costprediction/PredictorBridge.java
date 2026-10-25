package com.insurance.costprediction;

import android.content.Context;
import android.webkit.JavascriptInterface;

public class PredictorBridge {
    private final Context context;

    // Exact scikit-learn Lasso regression model parameters
    private static final double INTERCEPT = 14165.077267028533;
    
    private static final double[] COEF = {
        2791.8726072681425, // age
        310.39027331830977, // bmi
        209.12033956677337, // children
        6450.567814959236,  // smoker_bmi
        837.5415076361654,  // age_sq
        141.38803636269654, // sex_male
        8730.037229284908,  // smoker_yes
        375.14285221670997, // region_northwest
        402.7849246556818,  // region_southeast
        436.4935085544985   // region_southwest
    };

    private static final double[] SCALER_MEAN = {
        41.26635514018692,
        30.483018691588786,
        1.0728971962616822,
        6.347327102803739,
        1888.6906542056074
    };

    private static final double[] SCALER_SCALE = {
        13.63006190923491,
        6.2156536448949185,
        1.1777985240537394,
        12.674702877879877,
        1129.7094890009662
    };

    public PredictorBridge(Context context) {
        this.context = context;
    }

    @JavascriptInterface
    public String predict(int age, String sex, double bmi, int children, String smoker, String region) {
        double smokerBmi = "yes".equalsIgnoreCase(smoker) ? bmi : 0.0;
        double ageSq = (double) age * (double) age;

        double ageStd = (age - SCALER_MEAN[0]) / SCALER_SCALE[0];
        double bmiStd = (bmi - SCALER_MEAN[1]) / SCALER_SCALE[1];
        double childStd = (children - SCALER_MEAN[2]) / SCALER_SCALE[2];
        double smokerBmiStd = (smokerBmi - SCALER_MEAN[3]) / SCALER_SCALE[3];
        double ageSqStd = (ageSq - SCALER_MEAN[4]) / SCALER_SCALE[4];

        double sexMale = "male".equalsIgnoreCase(sex) ? 1.0 : 0.0;
        double smokerYes = "yes".equalsIgnoreCase(smoker) ? 1.0 : 0.0;
        double regNw = "northwest".equalsIgnoreCase(region) ? 1.0 : 0.0;
        double regSe = "southeast".equalsIgnoreCase(region) ? 1.0 : 0.0;
        double regSw = "southwest".equalsIgnoreCase(region) ? 1.0 : 0.0;

        double agePart = (COEF[0] * ageStd) + (COEF[4] * ageSqStd);
        double bmiPart = (COEF[1] * bmiStd);
        double childPart = (COEF[2] * childStd);
        double smokerPart = (COEF[3] * smokerBmiStd) + (COEF[6] * smokerYes);
        double regionPart = (COEF[7] * regNw) + (COEF[8] * regSe) + (COEF[9] * regSw);
        double sexPart = (COEF[5] * sexMale);

        double total = INTERCEPT + agePart + bmiPart + childPart + smokerPart + regionPart + sexPart;
        if (total < 0.0) total = 0.0;

        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"total\":").append(total).append(",");
        sb.append("\"base\":").append(INTERCEPT).append(",");
        sb.append("\"agePart\":").append(agePart).append(",");
        sb.append("\"bmiPart\":").append(bmiPart).append(",");
        sb.append("\"childPart\":").append(childPart).append(",");
        sb.append("\"smokerPart\":").append(smokerPart).append(",");
        sb.append("\"regionPart\":").append(regionPart);
        sb.append("}");

        return sb.toString();
    }
}
