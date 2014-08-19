define([
	"lib/d3.v3.min",
	"lib/underscore",
	"js/info_pane",
	"js/algo_detail",
	"js!lib/colorbrewer.js"],
	function(d3, _, pane, algo_detail) {

	var graph = {};

	var width = parseInt(d3.select("#graph").style("width")),
		height = parseInt(d3.select("#graph").style("max-height")) - 10;

	var svg = d3.select("#graph").append("svg")
		.attr("width", 0)
		.attr("height", 0);

	var nodes = [],
		links = [];

	var colors = function(e) {
		i = colorbrewer['YlGnBu'][4];
		return i[e]
	};

	var force = d3.layout.force()
		.size([width, height])
		.gravity(.15)
		.charge(-250)
		.linkDistance(function (l) { return 400*l.value; })
		.linkStrength(1);

	// Add data to graph
	graph.update = function(paper_id) {

		// Set the width/height of the SVG (initially set to 0)
		svg.attr("width", width)
			.attr("height", height);

		// Load data and start force layout
		d3.json("/d/" + paper_id + "-" + 10 + "/", function(error, neighbors_data) {

			graph_data = neighbors_data.graph_data
			neighbors_data = neighbors_data.neighbors_data

			// Initialize the description of the algorithm results
			algo_detail.display(neighbors_data)

			force
				.nodes(graph_data.nodes)
				.links(graph_data.links)
				.start();

			// Remove data from graph
			svg.selectAll(".link").remove();
			svg.selectAll(".node").remove();

			// Add data to the graph
			var node = svg.selectAll(".node")
				.data(force.nodes(), function(d){ return d.id; });
			var link = svg.selectAll(".link")
				.data(force.links(), function(d) { return d.source.id + "-" + d.target.id; });

			// Initialize links and nodes
			link.enter().append("line")
				.attr("class", "link")
			node.enter().append("circle")
				.attr("class", "node")
				.on("click", nodeClick)
				.attr("r", function(n) { return 15 - 2.5*n.level*n.level; })
				.attr("fill", function(n) { return d3.rgb(colors(n.level+1)); })
				.attr("stroke-width", "0")
				.call(force.drag);

			// Activate root node
			var rootNode = undefined;
			svg.selectAll(".node")
				.classed("active", function(n) { 
					if (n.level == 0) rooNode = n;
					return n.level == 0; });
			svg.selectAll(".link")
				.classed("active", function(p) { return p.source == rooNode || p.target == rooNode; });
			repaintGraph()

			// Display info panel for root node
			pane.display(rooNode);

			// Start force
			force.on("tick", tick);
			force.start();

			// add legend   
				var legend = svg.append("g")
				  .attr("class", "legend")
				  .attr("x", width - 65)
				  .attr("y", 25)
				  .attr("height", 100)
				  .attr("width", 100);

				legendText = [
						{
							"text": "Selected node", 
							"r": 15, 
							"fill": d3.rgb(colors(0)), 
							"strokewidth": 3
						},
						{
							"text": "Root node", 
							"r": 15, 
							"fill": d3.rgb(colors(1)), 
							"strokewidth": 0
						},
						{
							"text": "1st level neighbors", 
							"r": 12.5, 
							"fill": d3.rgb(colors(2)), 
							"strokewidth": 0
						},
						{
							"text": "2nd level neighbors", 
							"r": 5, 
							"fill": d3.rgb(colors(3)), 
							"strokewidth": 0
						}
					];

				legend.selectAll('g').data(legendText)
					.enter()
					.append('g')
					.each(function(d, i) {
						var g = d3.select(this);

						g.append("circle")
							.attr("cx", width - 175)
							.attr("cy", i*35 + 30)
							.attr("r", function(d) { return d.r; })
							.attr("fill", function(d) { return d.fill; })
							.attr("stroke-width", function(d) { return d.strokewidth; })
							.attr("stroke", "black");

						g.append("text")
							.attr("x", width - 150)
							.attr("y", i * 35 + 34)
							.style("fill", "black")
							.text(function(d) { return d.text; });
					});





			function tick() {
				link.attr("x1", function(d) { return d.source.x; })
					.attr("y1", function(d) { return d.source.y; })
					.attr("x2", function(d) { return d.target.x; })
					.attr("y2", function(d) { return d.target.y; });
				node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
			}

			// What happens when a node is clicked
			function nodeClick(d) {
				pane.display(d);							// Display information panel for this node
				deactivateAll();							// Reset graph (deactivate nodes/links and reset style)
				d3.select(this).classed("active", true);	// Activate this node
				svg.selectAll(".link")						// Activate its links
					.classed("active", function(p) { 
						return d3.select(this).classed("active") || p.source === d || p.target === d; 
					});
				repaintGraph();								// Repaint graph
			}

			// Deactive all nodes and links
			function deactivateAll() {
				svg.selectAll(".node").classed("active", false);
				svg.selectAll(".link").classed("active", false);
			}

			// Correctly repaint activated/deactivated nodes
			function repaintGraph() {
				// Repaint all links/nodes as if deactivated
				svg.selectAll(".link")
					.style("stroke-width", "1.5");
				svg.selectAll(".node")
					.attr("r", function(n) { return 15 - 2.5*n.level*n.level; })
					.style("fill", function(n) { return d3.rgb(colors(n.level+1)); })
					.style("stroke-width", "0");
				// Repaint activated links/nodes
				svg.selectAll(".link.active")
					.style("stroke-width", "5");
				svg.selectAll(".node.active")
					.style("fill", d3.rgb(colors(0)))
					.style("stroke-width", "3");
			}


		});
	}

	return graph;
})
