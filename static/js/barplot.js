define(["lib/d3.v3.min", "lib/nvd3/nv.d3.min"], function(d3) {

	barplot = {}

	barplot.set_graph = function(graph) {
		barplot.graph = graph;
	}

	barplot.update = function(data) {

		nv.addGraph(function() {
			var chart = nv.models.multiBarChart()
			  .stacked(true)
			  .showControls(false)
			  .reduceXTicks(false)   //If 'false', every single x-axis tick label will be rendered.
			  .rotateLabels(-30)      //Angle to rotate x-axis labels.
			  .groupSpacing(0.1)    //Distance between each group of bars.
			  .margin({top: 30, right: 20, bottom: 50, left: 175})
			  .tooltips(true)             //Show tooltips on hover
			  .tooltip(function(key, x, y, e, graph) {
			    return '<h4>' + key + '</h4>' +
			        	'<p>' +  y + ' on ' + x + '</p>';
			  });
			;

			chart.xAxis
			.tickFormat(function(d) {
				return data[0]['values'][d]['xlabel'];
			});

			chart.yAxis
			.tickFormat(d3.format(',.4f'));

			d3.select('#chart1 svg')
			.datum(data)
			.call(chart);

			nv.utils.windowResize(chart.update);

			return chart;
			
		},function(){
			  d3.selectAll(".nv-bar").on('click',
					function(d){
						barplot.graph.update(d['xlabel']);
			   });
		});

	}

	return barplot
});
