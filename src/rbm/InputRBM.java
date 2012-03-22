package rbm;

import java.util.Random;
import java.io.Serializable;

public class InputRBM extends SimpleRBM implements Serializable{

	private static final long serialVersionUID = 5035486678274163394L;
	
	Node[] visibleNodes;
    transient Random rand = new Random();

    /*
      * Constructor for InputRBM
      */
    public InputRBM(int numVisibleNodes, int numHiddenNodes){
        super(numVisibleNodes, numHiddenNodes);
        super.visibleNodes = null;   // we use our own, any use of the other
                                     // will be an error

        this.visibleNodes = new Node[numVisibleNodes + 1];
        for(int i = 0; i < visibleNodes.length; i++){
            visibleNodes[i] = new Node(false, false);
        }

        // bias nodes are always 1
        visibleNodes[ visibleNodes.length - 1 ].setValue(true);
        visibleNodes[ visibleNodes.length - 1 ].clamped = true;


    } // end of constructor

    /**
     *
     * @param startIndex - the index to start clamping at, inclusive
     * @param endIndex - the index to stop clamping at, exclusive
     *
     * Sets the nodes corresponding to the array of indices to clamped.
     * For example, giving it [7,9] would clamp the input nodes at index [7,9).
     *
     */
    public void clamp(int startIndex, int endIndex){
        for(int i = startIndex; i < endIndex; i++){
            if(i >= visibleNodes.length || i < 0)
                continue;

            visibleNodes[i].clamped = true;
        }
    } // end of method clamp


    /**
     *
     * @param index - the index to clamp
     *
     * Sets the clamp for a single index
     */
    public void clamp(int index){
        if(index >= visibleNodes.length)
            return;

        visibleNodes[index].clamped = true;
    } // end of method clamp

    /**
     *
     * @param startIndex - the index to start unclamping at, inclusive
     * @param endIndex - the index to stop unclamping at, exclusive
     *
     * Sets the nodes corresponding to the array of indices to unclamped.
     */
    public void unclamp(int startIndex, int endIndex){
        for(int i = startIndex; i < endIndex; i++){
            if(i >= visibleNodes.length || i < 0)
                continue;

            visibleNodes[i].clamped = false;
        }
    } // end of method clamp


    /**
     *
     * @param index - the index to unclamp
     *
     * Releases the clamp for a single index
     */
    public void unclamp(int index){
        if(index >= visibleNodes.length)
            return;

        visibleNodes[index].clamped = false;
    } // end of method clamp


    /**
     *
     * Unclamps all nodes in the visible layer.
     */
    public void unclampAll(){
        for(Node n: this.visibleNodes){
            n.clamped = false;
        }
    }


    /**
     * Probabilistically activates each node in the visible layer based upon
     * the weighted sum accumulated from the hidden layer.
     */
    @Override
    public void activateVisible()
    {
        for (int i=0; i<visibleNodes.length; ++i){
            if(visibleNodes[i].clamped == false){
                //computed weighted sum
                float sum = computeVisibleWeightedSum(i);

                //(call logsig function with annealing rate set to 1)
                // activate with that probability
               visibleNodes[i].setValue((this.rand.nextDouble() < logsig(sum, 1)));
       

            }
        }
    }

    /**
     * Probabilistically activates each node in the hidden layer based upon
     * the weighted sum accumulated from the visible layer.
     */
    @Override
    public void activateHidden()
    {
        for (int i=0; i<hiddenNodes.length-1; ++i)
        {
            //compute weighted sum
            float sum = 0;
            for (int j=0; j<visibleNodes.length; ++j)
                if (visibleNodes[j].value)
                    sum += weights[j][i];

            //Probabilistically activate node based on sigmoid computation
            hiddenNodes[i] = (rand.nextDouble() < logsig(sum, annealingRate));

        }
    }


    /**
     * For each group, probabilistically chooses one bit from that group to
     * activate based on the probability distribution defined by each bit's
     * probability of activation. Once one bit is chosen, all other bits in the
     * group are set to 0.
     */
   
    /**
     *
     * @param probabilities - the array of probabilities that define a distribution
     *
     * Helper method to normalize the probabilities and create a distribution for
     * the activateVisibleProbDist function.
     */
    @SuppressWarnings("unused")
	private static void normalizeProbabilities(float[] probabilities) {
        float sum = 0;
        for (int i = 0; i < probabilities.length; i++) {
            sum += probabilities[i];
        }
        //Normalize probabilities by dividing by sum
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = probabilities[i]/sum;
        }
        float lastValue = 0;
        //Create distribution - distance between each value in the array will
        //correspond to that bit's likelihood of being picked.
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] += lastValue;
            lastValue = probabilities[i];
        }
    }
    

 
    /**
     * Method: accumulatePos / accumulateNeg
     *
     * Increments positive/negative weight change matrices by product of input
     * node states and hidden node states
     *
     */
    @Override
    protected void accumulatePos()
    {
        for (int i=0; i<visibleNodes.length; ++i)
            if (visibleNodes[i].value)
                for (int j=0; j<hiddenNodes.length; ++j)
                    dPos[i][j] += hiddenNodes[j] ? 1 : 0;
    }

    @Override
    protected void accumulateNeg()
    {
        for (int i=0; i<visibleNodes.length; ++i)
            if (visibleNodes[i].value)
                for (int j=0; j<hiddenNodes.length; ++j)
                    dNeg[i][j] += hiddenNodes[j] ? 1 : 0;
    }

    /**
     *
     * @param newInput - the bit array to set as the visible layer
     */
    @Override
    public void setInput(boolean[] newInput){

        for (int i=0; i < newInput.length; i++)
        {
            //modify value directly (rather than using setValue)
            //in order to reset clamped nodes
            this.visibleNodes[i].value = newInput[i];
        }

    } // end of method setInput

    /**
     *
     * @return - an int[] copy of the state of the visible layer
     */
    @Override
    public boolean[] getVisible(){
        boolean[] visbools = new boolean[this.visibleNodes.length];

        for(int i = 0; i < this.visibleNodes.length; i++){
            visbools[i] = this.visibleNodes[i].value;
        }

        return visbools;
    }

    /**
     *
     * @param numRows - the number of rows in the diagram
     * @param numCols - the number of columns in the diagram
     * @return - all the activation probabilities for the visible nodes,
     * organized into a 2D array
     */
    //@Override
    public float[] predict() {
    	float currSum;
        float[] probs = new float[this.visibleNodes.length];
        for (int visIndex = 0; visIndex < probs.length; visIndex++) {
                currSum = computeVisibleWeightedSum(visIndex);
                probs[visIndex] = logsig(currSum, 1);
            }
        return probs;
    }

    /**
     *
     * @return - the current energy state of the machine
     */
    @Override
    public float getEnergy()
    {
    	int vn;
    	int hn;
        float energy = 0;
        for (int i=0; i<visibleNodes.length; ++i)
            for (int j=0; j<hiddenNodes.length; ++j){
            	vn = visibleNodes[i].value ? 1 : 0;
        		hn = hiddenNodes[j] ? 1 : 0;
                energy -= vn*hn*weights[i][j];
            }
        return energy;
    }


    // class for clamped, unclamped inputs
    // (basically a tuple)
    class Node{
        public boolean value;
        public boolean clamped;

        public Node(boolean value, boolean clamped){
            this.value = value;
            this.clamped = clamped;
        }

        public void setValue(boolean v){
            if(this.clamped == false)
                this.value = v;
        }

        @Override
        public String toString(){
            return "Value: "+value+" Clamped: "+clamped;
        }
    }  // end of class Node


} // end class inputRBM
