package parse;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.regex.Pattern;

public class ParseFile {
	private String filename;
	private Pattern pattern;
	
	public ParseFile(String filename){
		this.filename = filename;
		this.pattern = Pattern.compile(",\\s*");
	}
	
	
	/**
	 * Returns the data for task2.
	 * @return 
	 */
	public boolean[][] getData(){
		boolean[][] data = new boolean[countLines()][];
		String currentLine;
		int lineNumber;
		
		try{
			BufferedReader br = new BufferedReader(new FileReader(new File(this.filename)));
			currentLine = br.readLine();
			lineNumber = 0;
			String[] split;
			
			
			while(currentLine != null){
				split = this.pattern.split(currentLine);
				data[lineNumber] = new boolean[split.length];
				
				for(int i = 0; i < split.length; i++ ){
					data[lineNumber][i] = split[i].equals(1);
				}
				
				lineNumber += 1;
				currentLine = br.readLine();
			}
			
		}catch(Exception e){
			e.printStackTrace();
		}
		
		
		return data;
	} // end of method getData
	
	
	/**
	 * 
	 * @param filename
	 * @return array of layer sizes
	 */
	public int[] getSizes(String filename){
		int lines = 0;
		
	    int[] sizes = null;
		try{
			// get line count
	    	BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
	    	while(br.readLine() != null){
	    		lines += 1;
	    	}
	    	br.close();
	    	
	    	// make an array of the appropriate size, populate it
	    	sizes = new int[lines];
	    	br = new BufferedReader(new FileReader(new File(filename)));
	    	for(int i = 0; i < lines; i++){
	    		sizes[i] = Integer.parseInt(br.readLine());
	    		
	    	}
	    	
	    	
	    	
	    }catch(Exception e){
	    	e.printStackTrace();
	    }
	    
	    
		return sizes;
	} // end of method getSizes
	
	
	/**
	 * 
	 * @return the number of lines in the file passed to the constructor.
	 * Doesn't yet check to see if the last line is empty.
	 */
	public int countLines(){
		int lines = 0;
		
		try{
			BufferedReader br = new BufferedReader(new FileReader(new File(this.filename)));
			while(br.readLine() != null){
				lines += 1;
			}
			
			br.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		return lines;
	} // end of method countLines
	
	
} // end of class ParseFile
