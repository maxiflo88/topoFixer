/*
 * Created by SharpDevelop.
 * User: edgars
 * Date: 7/30/2021
 * Time: 11:52
 * 
 * To change this template use Tools | Options | Coding | Edit Standard Headers.
 */
using System;
using System.IO;
using Autodesk.Revit.UI;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI.Selection;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using Autodesk.Revit.DB.Architecture;

namespace fixTopo
{
    [Autodesk.Revit.Attributes.Transaction(Autodesk.Revit.Attributes.TransactionMode.Manual)]
    [Autodesk.Revit.DB.Macros.AddInId("C18E33E9-591D-4AB4-ABC8-C980AE758BB5")]
	public partial class ThisApplication
	{
		private void Module_Startup(object sender, EventArgs e)
		{

		}

		private void Module_Shutdown(object sender, EventArgs e)
		{

		}
				const double _feet_to_mm = 25.4 * 12;
		
		static int ConvertFeetToMillimetres( double d )
		{
		  return (int) ( _feet_to_mm * d + 0.5 );
		}

	    class MyFailureProcessor : IFailuresPreprocessor
	    {
	        public FailureProcessingResult PreprocessFailures(FailuresAccessor failuresAccessor)
	        {
	            // TODO: implement code
	            return FailureProcessingResult.Continue;
	        }
	    }
		private void createTopo(string filename, TopographySurface surface, Document doc){
			IList<XYZ>points = surface.GetPoints();
			int count=0;
			//start editscope
	        TopographyEditScope scope = new TopographyEditScope(doc, "change topo");
	        if(scope.IsActive){
	        	scope.Cancel();
	        }
			scope.Start(surface.Id);
	        using (Transaction t = new Transaction(doc, "create topo"))
        	{
		        t.Start();
		        foreach(string line in File.ReadAllLines(filename)){
					string[] info=line.Split(',');
					int changes=Int32.Parse(info[1]);
					double elevation=UnitUtils.ConvertToInternalUnits(double.Parse(info[0]), DisplayUnitType.DUT_MILLIMETERS);
					if(changes==3){
//						surface.MovePoint(points[count], new XYZ(points[count].X, points[count].Y, elevation));
//						surface.MovePoints(new List<XYZ>(){points[count]}, new XYZ(0, 0, elevation-points[count].Z));
						surface.ChangePointElevation(points[count], elevation);
					}
					else if(changes==1){
						surface.DeletePoints(new List<XYZ>(){points[count]});
					}
					count++;
				}
		        t.Commit();
	        }
	        MyFailureProcessor failProc = new MyFailureProcessor();
		    scope.Commit(failProc);
		}
		private void runPythonScript(string dir)
		{
			
			var process = new Process();
			process.StartInfo.FileName=Path.Combine(dir, "calculatePoints.exe");
			process.StartInfo.WorkingDirectory=dir;
			
			process.Start();
			process.WaitForExit();
			
		}
		private void saveTopoTXT(TopographySurface surface, string filename){
			List<string>text=new List<string>();
			foreach(XYZ point in surface.GetPoints()){
				if(surface.IsBoundaryPoint(point)){
					text.Add(string.Format("{0} {1} {2} 1", ConvertFeetToMillimetres( point.X ), 
									        	                       		ConvertFeetToMillimetres( point.Y ), 
									        	                       		ConvertFeetToMillimetres( point.Z )));
				}
				else{
					text.Add(string.Format("{0} {1} {2} 0", ConvertFeetToMillimetres( point.X ), 
									        	                       		ConvertFeetToMillimetres( point.Y ), 
									        	                       		ConvertFeetToMillimetres( point.Z )));
				}
			}
			if(text.Count!=0) File.WriteAllLines(filename, text);
		}
		#region Revit Macros generated code
		private void InternalStartup()
		{
			this.Startup += new System.EventHandler(Module_Startup);
			this.Shutdown += new System.EventHandler(Module_Shutdown);
		}
		#endregion
		public void topo_main()
		{
			string dir=@"G:\revit projects\topoProject\topoFixer\output";
			Document doc =ActiveUIDocument.Document;
			ElementId ele= ActiveUIDocument.Selection.GetElementIds().FirstOrDefault();
			TopographySurface surface = doc.GetElement(ele) as TopographySurface;
			saveTopoTXT(surface, Path.Combine(dir, "topo.txt"));
			runPythonScript(dir);
			createTopo(Path.Combine(dir, "result.txt"), surface, doc);
		}
	}
}