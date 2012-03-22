package main;

import parse.ParseFile;
import rbm.SimpleLayeredRBM;
import java.io.ObjectOutputStream;
import java.io.FileOutputStream;
import java.io.File;

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		if(args.length != 5){
			System.out.println("Usage:\n\tTraining set name\n\tTest set name\n\tLayer File Name\n\tPrediction output name\n\tserialized file name");
			System.exit(0);
		}
		
		String trainingName = args[0]; 
		String testName = args[1];
		String layerFileName = args[2];
		String predictOutName = args[3];
		String serialName = args[4];
		
		System.out.println("Training on " + trainingName);
		ParseFile trainParse = new ParseFile(trainingName);
		
		boolean[][] trainingData = trainParse.getData();
		int sizes[] = trainParse.getSizes(layerFileName);
		SimpleLayeredRBM slrbm = new SimpleLayeredRBM(trainingData[0].length, sizes);
		
		slrbm.train(trainingData, 100);
		System.out.println("Done training!  Now to serialize the file:" + serialName);
		try{
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(new File(serialName)));
			out.writeObject(slrbm);
			out.close();
		}catch(Exception e){
			e.printStackTrace();
		}

		System.out.println("Done serializing!  Now to make some predictions on " + testName);
		System.out.println("Outputting predictions to " + predictOutName);
		
		
		ParseFile testParse = new ParseFile(testName);
		boolean[][] testData = testParse.getData();
		
		slrbm.predict(testData, 25, predictOutName);
		
		System.out.println("Done!  I hope you win!");
		
	} // end of main method

} // end of class main
