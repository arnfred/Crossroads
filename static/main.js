var graph = function () {
	var nodes = d3.range(100).map(function (i) { return { index: i } });
	var links = nodes.map(function (i,n) { 
		return { 
			target: i, 
			source: Math.floor(Math.random()*100),
			value: Math.floor(Math.random()*100)
		} 
	})
	return { nodes : nodes, links : links }
}()

var width = 960,
    height = 500

var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height);

var force = d3.layout.force()
    .gravity(.15)
    .distance(100)
    .charge(-100)
	//.linkDistance(function (l) { return l.value; })
	.linkStrength(function (l) { return l.value / 100.0; })
	.size([width, height]);

// Add data to graph
var start = function(id) {
	// Stop and clear existing force layout
	svg.selectAll(".link").remove()
	svg.selectAll(".node").remove()

	// Load data and start force layout
	d3.json("/d/" + id, function(error, json) {

		force
			.nodes(graph.nodes)
			.links(graph.links)
			.start();

		var link = svg.selectAll(".link")
			.data(graph.links)
			.enter().append("line")
			.attr("class", "link");

		var node = svg.selectAll(".node")
			.data(graph.nodes)
			.enter().append("circle")
			.attr("class", "node")
			.attr("r", 5)
			.call(force.drag);

		force.on("tick", function() {
		  link.attr("x1", function(d) { return d.source.x; })
			  .attr("y1", function(d) { return d.source.y; })
			  .attr("x2", function(d) { return d.target.x; })
			  .attr("y2", function(d) { return d.target.y; });

		  node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
		});
	});
}

// Start without id
start()
