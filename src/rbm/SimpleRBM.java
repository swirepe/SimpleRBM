package rbm;

import java.util.Random;
import java.io.Serializable;

public class SimpleRBM implements Serializable {
	  //member variables

    /**
	 * 
	 */
	private static final long serialVersionUID = -4412383722010406930L;
	
	protected boolean[] visibleNodes;
	protected boolean[] hiddenNodes;
	protected float[][] weights;
	protected float[][] dPos;    // accumulates positive weight changes
	protected float[][] dNeg;    // accumulates negative weight changes
	protected float annealingRate; // multiplier in sigmoid function
    
    protected transient Random rand = new Random();
    
    /*
     *  Method: constructor
     *
     *  constructs a new RBM
     */
    public SimpleRBM(int numVisibleNodes, int numHiddenNodes) {
        //initialize nodes
        this.visibleNodes = new boolean[numVisibleNodes + 1]; //add one spot for bias
        this.hiddenNodes = new boolean[numHiddenNodes + 1];
        visibleNodes[visibleNodes.length - 1] = true; //bias is always on
        hiddenNodes[hiddenNodes.length - 1] = true;

        //initialize weights and weight change matrices
        this.weights = new float[numVisibleNodes + 1][numHiddenNodes + 1];
        this.dPos = new float[numVisibleNodes + 1][numHiddenNodes + 1];
        this.dNeg = new float[numVisibleNodes + 1][numHiddenNodes + 1];
         //randomly initialize weights
        for (int i = 0; i < numVisibleNodes + 1; ++i) {
            for (int j = 0; j < numHiddenNodes + 1; ++j) {
                weights[i][j] = new Float(0.1 * rand.nextGaussian());
                dPos[i][j] = 0;
                dNeg[i][j] = 0;
            }
        }
    }

    /*
     * Method: constructor
     *
     * constructs a new SimpleRBM using a pre-defined set of input nodes.
     * useful when layering RBMs.
     */
    public SimpleRBM(boolean[] visibleNodes, int numHiddenNodes) {
        this.visibleNodes = visibleNodes;

        this.hiddenNodes = new boolean[numHiddenNodes + 1];
        this.hiddenNodes[hiddenNodes.length - 1] = true;

        //initialize weights and weight change matrices
        this.weights = new float[visibleNodes.length][numHiddenNodes + 1];
        this.dPos = new float[visibleNodes.length][numHiddenNodes + 1];
        this.dNeg = new float[visibleNodes.length][numHiddenNodes + 1];
         //randomly initialize weights
        for (int i = 0; i < visibleNodes.length; ++i) {
            for (int j = 0; j < numHiddenNodes + 1; ++j) {
                weights[i][j] = new Float(0.1 * rand.nextGaussian());
                dPos[i][j] = 0;
                dNeg[i][j] = 0;
            }
        }
    }

    public boolean[] getVisible() {
        return visibleNodes;
    }

    /*
     * Method: setInput
     *
     * To be called only in LayeredRBM for first layer
     */
    public void setInput(boolean[] newInput) {
        for (int i = 0; i < visibleNodes.length - 1 && i < newInput.length; i++) {
            visibleNodes[i] = newInput[i];
        }
    }

    public boolean[] getHidden() {
        return hiddenNodes;
    }

    public float[][] getWeights() {
        return weights;
    }

    public void setWeights(float[][] weights) {
        this.weights = weights;
    }

    public void setAnnealingRate(float newRate) {
        annealingRate = newRate;
    }

    /*
     *  Method: train
     *
     *  Trains the network on a single input (assumes input has already
     *  been set) and adds initial and final contrastive divergences to accumulators
     *
     */
    public void train(int numCycles) {
        activateHidden();
        accumulatePos();
        for (int i = 0; i < numCycles; ++i) {
            activateVisible();
            activateHidden();
        }
        accumulateNeg();
    }

    /*
     *  Method: activateVisible / activateHidden
     *
     *  Chooses whether to activate each input/hidden node based on the
     *  activation states and weights of the other nodes
     */
    public void activateVisible() {

        //initialize random number generator
        

        for (int i = 0; i < visibleNodes.length - 1; ++i) {
            //compute weighted sum
            float sum = computeVisibleWeightedSum(i);
            // Probabilistically activate node based on sigmoid computation
            visibleNodes[i] = (rand.nextDouble() < logsig(sum, 1));
        }
    }

    public void activateHidden() {
        
        for (int i = 0; i < hiddenNodes.length - 1; ++i) {
            //compute weighted sum
            float sum = 0;
            for (int j = 0; j < visibleNodes.length; ++j) {
                if (visibleNodes[j]) {
                    sum += weights[j][i];
                }
            }

            // Probabilistically activate node based on sigmoid computation
            hiddenNodes[i] = (rand.nextDouble() < logsig(sum, annealingRate));
        }
    } // end of method activate hidden

    
    //Computes the weighted sum for a visible node
    public float computeVisibleWeightedSum(int index) {
        float sum = 0;
        for (int j = 0; j < hiddenNodes.length; j++) {
            if (hiddenNodes[j]) {
                sum += weights[index][j];
            }
        }
        return sum;
    }

    /*
     * Method: accumulatePos / accumulateNeg
     *
     * Increments positive/negative weight change matrices by product of input
     * node states and hidden node states
     *
     */
    protected void accumulatePos() {
        for (int i = 0; i < visibleNodes.length; ++i) {
            if (visibleNodes[i]) {
                for (int j = 0; j < hiddenNodes.length; ++j) {
                    dPos[i][j] += hiddenNodes[j] ? 1: 0;
                }
            }
        }
    }

    protected void accumulateNeg() {
        for (int i = 0; i < visibleNodes.length; ++i) {
            if (visibleNodes[i]) {
                for (int j = 0; j < hiddenNodes.length; ++j) {
                    dNeg[i][j] += hiddenNodes[j] ? 1 : 0;
                }
            }
        }
    }


    /*
     * Method: updateWeights
     *
     * increments/decrements weights by positive and negative weight change
     * matrices respectively (divided by the total number of inputs).  Also
     * resets weight change matrices.
     *
     */
    public void updateWeights(int numInputs) {
        for (int i = 0; i < weights.length; ++i) {
            for (int j = 0; j < weights[0].length; ++j) {
                weights[i][j] += (0.2f * dPos[i][j] / numInputs);
                weights[i][j] -= (0.2f * dNeg[i][j] / numInputs);

                dPos[i][j] = 0;
                dNeg[i][j] = 0;
            }
        }
    }

    /*
     * Method: getEnergy
     *
     * returns the the current energy state of the network, computed by summing
     * all weights between active nodes
     *
     */
    public float getEnergy() {
        float energy = 0;
        int vn;
        int hn;
        for (int i = 0; i < visibleNodes.length; ++i) {
            for (int j = 0; j < hiddenNodes.length; ++j) {
            	vn = visibleNodes[i] ? 1 : 0;
            	hn = hiddenNodes[j] ? 1 : 0;      			
                energy -= vn * hn * weights[i][j];
            }
        }
        return energy;
    }

    /*
     * Method: logsig
     *
     * computes the logsig of the input number
     *
     * logsig is a sigmoidal function with range between 0 and 1
     *
     * input is adjusted by annealing rate, which should be between 0 and 1
     *
     */
    protected static float logsig(float x, float annealingRate) {
        return 1 / (1 + ((float) Math.exp(-x / annealingRate)));
    }
}
