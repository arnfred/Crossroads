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

		// Node selected to show info panel (init root node as selected node)
		var selected_node_id = paper_id;
		var selected_node_level = 0;
		var selected_node = undefined;
		var selected_circle = undefined;
		var selected_circle_color = d3.rgb(colors(0));
		var selected_node_stroke_width = "3";

		svg.attr("width", width)
			.attr("height", height);

		// Load data and start force layout
		d3.json("/d/" + paper_id + "-" + 10 + "/", function(error, neighbors_data) {

			graph_data = neighbors_data.graph_data
			neighbors_data = neighbors_data.neighbors_data

			// Mapping between paper ids and array ids
			graph_idx = _.map(graph_data.nodes, function(n){return n.id});
			nodes_idx = _.map(nodes, function(n){return n.id});

			// Find center node id
			center_node_id = _.indexOf(graph_idx, paper_id)

			// Initialize info panel with center node data
			focused_node = graph_data.nodes[center_node_id];
			pane.display(focused_node);

			// Initialize the description of the algorithm results
			algo_detail.display(neighbors_data)

			var tmp_nodes = [],
				tmp_links = [];

			svg.selectAll(".link").remove();
			svg.selectAll(".node").remove();

			// Iterate over new graph nodes
			for (var i in graph_data.nodes) {
				var graph_node = graph_data.nodes[i];

				// Get id of current node in graph
				var current_id = _.indexOf(nodes_idx, graph_node.id);

				// If this node is new
				if (current_id == -1) {
					// Get id of parent node in graph
					var parent_id = _.indexOf(nodes_idx, graph_node.parent_id);
					if (parent_id != -1) {
						var parent_node = nodes[parent_id];
						graph_node.x = parent_node.x + _.random(-10, 10);
						graph_node.y = parent_node.y + _.random(-10, 10);
						graph_node.px = parent_node.px + _.random(-10, 10) ;
						graph_node.py = parent_node.py + _.random(-10, 10);
					}
					// Add it to the graph
					tmp_nodes.push(graph_node);
				}
				// If this node already exists
				else {
					// Keep track of its position
					var current_node = nodes[current_id];
					graph_node.x = current_node.x;
					graph_node.y = current_node.y;
					graph_node.px = current_node.px;
					graph_node.py = current_node.py;
					tmp_nodes.push(graph_node);
				}
			}
			for (var i in graph_data.links) {
				idx = links.indexOf(graph_data.links[i]);
				if (idx == -1) {
					tmp_links.push(graph_data.links[i]);
				}
			}

			nodes = tmp_nodes;
			links = tmp_links;

			force
				.nodes(nodes)
				.links(links)
				.start();

			var node = svg.selectAll(".node")
				.data(force.nodes(), function(d){
					return d.id;
				});

			var link = svg.selectAll(".link")
				.data(force.links(), function(d) {
					return d.source.id + "-" + d.target.id;
				});

			link.enter().append("line")
				.attr("class", "link");
			link.exit().remove();

			node.enter().append("circle")
				.attr("class", "node")
				.on("click", function(n) {
					pane.display(n);
					if (selected_node != undefined) {
						d3.select(selected_node).style("fill", d3.rgb(colors(selected_node_level+1)));
						d3.select(selected_node).style("stroke-width", "0");
					}
					selected_node = this;
					selected_node_level = n.level;
					d3.select(selected_node).style("fill", selected_circle_color);
					d3.select(selected_node).style("stroke-width", selected_node_stroke_width);
				})
				.attr("r", function(n) {
					return 15 - 2.5*n.level*n.level;
				})
				.attr("fill", function(n) {
					if (n.level == 0) {
						// set the selected_node as this node (Kind of hacky way to do it, TO FIX)
						selected_node = this;
						return selected_circle_color;
					} else {
						return  d3.rgb(colors(n.level+1));
					}
				}).
				attr("stroke-width", function(n) {

					if (n.level == 0) {
						console.log(selected_node)
						return selected_node_stroke_width;
					} else {
						return  "0";
					}
				})
				.call(force.drag);
			node.exit().remove();
			force.on("tick", tick);
			force.start();


			function tick() {
				link.attr("x1", function(d) { return d.source.x; })
					.attr("y1", function(d) { return d.source.y; })
					.attr("x2", function(d) { return d.target.x; })
					.attr("y2", function(d) { return d.target.y; });
				node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
			}

		});
	}

	return graph;
})
