package rbm;

import java.util.Random;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.File;
import java.io.Serializable;

public class SimpleLayeredRBM implements Serializable {

	private static final long serialVersionUID = -8685661487940297519L;
	
	//member variables
    protected SimpleRBM[] layers;
    protected transient Random rand = new Random();
    
    /**
     *
     * @param inputLength - the length of the inputs to this lrbm
     * @param layerSizes - an integer array of the number of hidden nodes in
     *                     each layer of this lrbm
     *
     * Constructs a new SimpleLayeredRBM given the above parameters
     */
    public SimpleLayeredRBM(int inputLength, int[] layerSizes){
        layers = new SimpleRBM[layerSizes.length];

        layers[0] = new InputRBM(inputLength, layerSizes[0]);
        for (int i = 1; i < layerSizes.length; i++) {
            layers[i] = new SimpleRBM(layers[i - 1].getHidden(), layerSizes[i]);
        }
    }

    
    /**
     * layeredLearn
     * @param inputs - the array of int arrays to train on
     *
     * Trains layered RBM on a series of input arrays by training each RBM layer
     * in turn.  Layers are trained through the contrastive divergence method
     * implemented in RBM.train().  For each RBM layer after the first, inputs
     * are propagated through previous layers by repeatedly activating hidden
     * nodes.
     */
    public void layeredLearn(boolean[][] inputs, int numEpochs) {

        for (int currLayer = 0; currLayer < layers.length; currLayer++) {
            for (int epoch = 0; epoch < numEpochs; epoch++) {
                //set annealing rate (falls from 1 to 0 during training)
                float annealingRate = 1 - (1f/numEpochs)*epoch;
                layers[currLayer].setAnnealingRate(annealingRate);

                for (int currInput = 0; currInput < inputs.length; currInput++){
                        propagateInput(inputs[currInput], currLayer);
                        layers[currLayer].train(5);
                        layers[currLayer].updateWeights(inputs.length);
                }
            }
        }
    } // end of method layeredLearn

    
    
    public void train(boolean[][] inputs, int numEpochs){
    	((InputRBM)this.layers[0]).clamp(0);
    	layeredLearn(inputs, numEpochs);
    	
    } // end of method train
    
    
    public void predict(boolean[][] testInputs, int numCycles, String outputFileName){
    	((InputRBM)this.layers[0]).unclamp(0);
    	
    	try{
    		
    	
	    	for(int i = 1; i < this.layers[0].getVisible().length; i++){
	    		((InputRBM)this.layers[0]).clamp(i);
	    	}
	    	
	    	boolean[] seed;
	    	for(boolean[] datapoint : testInputs){
	    		seed = new boolean[datapoint.length + 1];
	    		
	    		// start with a random prediction, fill in the with our observation
	    		seed[0] = this.rand.nextDouble() < 0.5;
	    		System.arraycopy(datapoint, 0, seed, 1, datapoint.length);
	    		
	    		layeredGenerate(seed, numCycles);	    		
	    	}
	    	
	    	float[] prediction = ((InputRBM)this.layers[0]).predict();
	    	
	    	
	    	// write the prediction to file
	    	BufferedWriter bw = new BufferedWriter(new FileWriter(new File(outputFileName)));
	    	for(float p: prediction){
	    		bw.write("" + p + "\n");
	    	}
	    	
	    	bw.close();
    	}catch(Exception e){
    		e.printStackTrace();
    	}
    	
    } // end of method predict
    
    /**
     *
     * @param seed - the integer array seed to generate from
     * @param numCycles - the number of activation cycles to perform
     * @return a new integer array generated from the given seed
     */
    public boolean[] layeredGenerate(boolean[] seed, int numCycles) {

        layers[0].setInput(seed);
        for (int cycle = 0; cycle < numCycles; cycle++)
        {
            // Propagate inputs forward through hidden layers
            for (int i = 0; i < layers.length; ++i) {
                layers[i].activateHidden();
            }

            // Propagate inputs backwards through visible layers
            for (int i = layers.length; i > 0; --i) {
                layers[i - 1].activateVisible();
            }
        }

        return layers[0].getVisible();
    }

    /**
     * propagateInput
     * @param input - the input to propagate through the network
     * @param depth - the number of layers to propagate through
     * @return
     *
     * Assigns input array to nodes of first network layer.  Propagates that
     * input through the network by repeatedly activating hidden nodes and
     * assigning the result to the next layers input nodes.
     */
    private void propagateInput(boolean[] input, int depth) {
        layers[0].setInput(input);
        for (int i = 0; i < depth; i++) {
            layers[i].activateHidden();
        }
    }
    
    
} // end of class SimpleLayeredRBM
