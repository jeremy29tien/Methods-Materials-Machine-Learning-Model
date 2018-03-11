import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.net.URL;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.io.File;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.awt.Desktop;

import org.jsoup.*;
import org.jsoup.nodes.*;
import org.jsoup.select.*;
import org.jsoup.Jsoup;
import org.jsoup.helper.Validate;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

public class MethodExtractor {
	
	private File cultureDir = new File("/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/cell-culture");
	private File imagingDir = new File("/Users/JeremyTien/Desktop/Machine-learning model/M&M-Data/cell-imaging");
	
	private int cultureCounter = 703; // currently has up to 1001, latest journal v.32(12) 2012 Jun
	private int imagingCounter = 698;
	
	public static void main (String[]args)
	{
		MethodExtractor me = new MethodExtractor();
		me.run();
	}
	
	public void run()
	{
		
		System.out.println("Drawing from Molecular and Cell Biology Journal archives on https://www.ncbi.nlm.nih.gov/pmc/journals/91/");
		
		String baseURL = "https://www.ncbi.nlm.nih.gov/pmc/journals/91/";
		Document basePage = null;
		try {
			basePage = Jsoup.connect(baseURL).get();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


        Elements links = basePage.select("a[href]");
       
        /*
        print("\nLinks: (%d)", links.size());
        for (Element link : links) {
        	String volumeURL = link.attr("abs:href");
            print(" * a: <%s>  (%s)", link.attr("abs:href"), trim(link.text(), 35));
            if(link.text().startsWith("v."))
            {
            	Document volumePage = null;
            	try {
        			volumePage = Jsoup.connect(volumeURL).get();
        		} catch (IOException e) {
        			// TODO Auto-generated catch block
        			e.printStackTrace();
        		}
            	Elements articleLinks = volumePage.select("a[href]");
            	for(Element articleLink : articleLinks)
            	{
            		String articleURL = articleLink.attr("abs:href");
            		print("\t\t * a: <%s>  (%s)", articleLink.attr("abs:href"), trim(articleLink.text(), 35));
            		if(articleLink.text().equals("PubMed"))
            		{
            			Document abstractPage = null;
            			try {
            				abstractPage = Jsoup.connect(articleURL).get();
                		} catch (IOException e) {
                			// TODO Auto-generated catch block
                			e.printStackTrace();
                		}
            			Elements fullTextLinks = abstractPage.select("a[href]");
            			
            			for(Element fullTextLink : fullTextLinks)
            			{
            				String fullTextURL = fullTextLink.attr("abs:href");
            				print("\t\t\t\t * a: <%s>  (%s)", fullTextLink.attr("abs:href"), trim(fullTextLink.text(), 35));
            				if(fullTextLink.text().contains("HighWire"))
            				{
            					Document target = null;
                    			try {
                    				target = Jsoup.connect(fullTextURL).get();
                        		} catch (IOException e) {
                        			// TODO Auto-generated catch block
                        			e.printStackTrace();
                        		}
                    			
                    			Elements mSection = target.getElementsByClass("section materials-methods");
                    			Elements paragraphs = mSection.select(".subsection");
                    			
                				
                    			for(Element paragraph : paragraphs)
                    			{
                    				String methodPara = paragraph.text();
                    				Scanner commandline = new Scanner(System.in);
                    				
                    				String subTitle = methodPara.substring(0, methodPara.indexOf(".")).toLowerCase();
                    				methodPara = methodPara.substring(methodPara.indexOf(".")+1);
                    				
                    				if(subTitle.contains("culture"))
                    				{
                    					System.out.println(subTitle.toUpperCase() + "\t\t\t" + methodPara + "\n");
                    					//System.out.println("\nProbably about cell culture, growth, or treatment. Y to confirm, N to reject.");
                    					String in = "Y"; //commandline.nextLine()
                    					if(in.equalsIgnoreCase("Y"))
                    					{
                    						File mfile = new File(cultureDir, cultureCounter + ".txt");
                    						cultureCounter++;
                    						try {
												PrintWriter writeToFile = new PrintWriter(mfile);
												writeToFile.println(methodPara);
												writeToFile.close();
											} catch (FileNotFoundException e) {
												// TODO Auto-generated catch block
												e.printStackTrace();
											}
                    					}
                    				}
                    				/*
                    				if(subTitle.contains("immuno") || subTitle.contains("stain") || subTitle.contains("analy"))
                    				{
                    					System.out.println(subTitle.toUpperCase() + "\t\t\t" + methodPara + "\n");
                    					System.out.println("\nProbably about image analysis, immunofluorescence, and staining. Y to confirm, N to reject.");
                    					String in = commandline.nextLine();
                    					if(in.equalsIgnoreCase("Y"))
                    					{
                    						File mfile = new File(imagingDir, imagingCounter + ".txt");
                    						imagingCounter++;
                    						try {
												PrintWriter writeToFile = new PrintWriter(mfile);
												writeToFile.println(methodPara);
												writeToFile.close();
											} catch (FileNotFoundException e) {
												// TODO Auto-generated catch block
												e.printStackTrace();
											}
                    					}
                    				}
                    				*
                    			}
            				}
            			}
            		}
            	}
            }
        }*/
        boolean contFromStop = false;
        print("\nLinks: (%d)", links.size());
        for (Element link : links) {
        	String volumeURL = link.attr("abs:href");
            //print(" * a: <%s>  (%s)", link.attr("abs:href"), trim(link.text(), 35));
        	
        	if(link.text().contains("v.35(5) 2015 Mar"))
        		contFromStop = true;
            if(link.text().startsWith("v.") && contFromStop)
            {
            	print(" * a: <%s>  (%s)", link.attr("abs:href"), trim(link.text(), 35));
            	Document volumePage = null;
            	try {
        			volumePage = Jsoup.connect(volumeURL).get();
        		} catch (IOException e) {
        			// TODO Auto-generated catch block
        			e.printStackTrace();
        		}
            	Elements articleLinks = volumePage.select("a[href]");
            	for(Element articleLink : articleLinks)
            	{
            		String articleURL = articleLink.attr("abs:href");
            		//print("\t\t * a: <%s>  (%s)", articleLink.attr("abs:href"), trim(articleLink.text(), 35));
            		if(articleLink.text().equals("Article"))
            		{
            			/*
            			//delay each time article is accessed to avoid being blocked
                		try {
                		    //Thread.sleep(2000);                 //1000 milliseconds is one second.
                			TimeUnit.SECONDS.sleep(2);
                		} catch(InterruptedException ex) {
                			System.out.println("Wait time interrupted.");
                		    Thread.currentThread().interrupt();
                		}
                		*/
                		
            			print("\t\t * a: <%s>  (%s)", articleLink.attr("abs:href"), trim(articleLink.text(), 35));
            			Document target  = null;
            			try {
            				target = Jsoup.connect(articleURL).get();
                		} catch (IOException e) {
                			// TODO Auto-generated catch block
                			e.printStackTrace();
                		}
            			
            			Elements sections = target.getElementsByClass("tsec sec");
            			for(Element section : sections)
            			{
            				//System.out.println(section.text()); // works
            				if(section.text().startsWith("MATERIALS AND METHODS"))
            				{
            					Elements paragraphs = section.select(".sec.sec-first, .sec, .sec.sec-last");
            					paragraphs.remove(0);
            					Elements subheaders = section.select(".inline");
            					int index = 0;
                    			for(Element paragraph : paragraphs)
                    			{
                    				String subTitle = subheaders.get(index).text().toLowerCase();
                    				String methodPara = paragraph.text();
                    				//System.out.println(paragraph.text()); //subheaders.get(index).text() + 
                    				/*
                    				if(subTitle.contains("culture"))
                    				{
                    					methodPara = methodPara.substring(methodPara.indexOf('.')+2);
                    					System.out.println(cultureCounter + ": " + subTitle.toUpperCase() + "\t\t\t" + methodPara + "\n");
                    					
                    					String in = "Y"; //commandline.nextLine()
                    					if(in.equalsIgnoreCase("Y"))
                    					{
                    						File mfile = new File(cultureDir, cultureCounter + ".txt");
                    						cultureCounter++;
                    						try {
												PrintWriter writeToFile = new PrintWriter(mfile);
												writeToFile.println(methodPara);
												writeToFile.close();
											} catch (FileNotFoundException e) {
												// TODO Auto-generated catch block
												e.printStackTrace();
											}
                    					}
                    				}
                    				*/
                    				
                    				if(subTitle.contains("immuno") || subTitle.contains("stain") || subTitle.contains("blot"))
                    				{
                    					String in = "Y";
                    					if(in.equalsIgnoreCase("Y"))
                    					{
                        					methodPara = methodPara.substring(methodPara.indexOf('.')+2);
                        					System.out.println(imagingCounter + ": " + subTitle.toUpperCase() + "\t\t\t" + methodPara + "\n");
                        					
                    						File mfile = new File(imagingDir, imagingCounter + ".txt");
                    						imagingCounter++;
                    						try {
												PrintWriter writeToFile = new PrintWriter(mfile);
												writeToFile.println(methodPara);
												writeToFile.close();
											} catch (FileNotFoundException e) {
												// TODO Auto-generated catch block
												e.printStackTrace();
											}
                    					}
                    				}
                    				
                    				index++; // advances subTitle list
                    			}
            				}
            			}
            			
            		}
            	}
            }
        }
	}
	private static void print(String msg, Object... args) {
	    System.out.println(String.format(msg, args));
	}
	
	private static String trim(String s, int width) {
	    if (s.length() > width)
	        return s.substring(0, width-1) + ".";
	    else
	        return s;
	}

}
