define(["lib/d3.v3.min", "lib/jquery"], function(d3, $) {

    var graph = {};

    var width = $("body").width();
        height = $("body").height();

    graph.svg = d3.select("body").append("svg")
        .attr("width", width)
        .attr("height", height);

    graph.force = d3.layout.force()
        .gravity(.15)
        .distance(10)
        .charge(-150)
        .linkDistance(function (l) { return l.value; })
        .linkStrength(function (l) { return l.value / 10.0; })
        .size([width, height]);

    // Add data to graph
    graph.start = function(id) {

        if (id == undefined) {
            id = "1304.5220"
        }
        // Stop and clear existing force layout
        graph.svg.selectAll(".link").remove()
        graph.svg.selectAll(".node").remove()

        console.debug(id)

        // Load data and start force layout
        d3.json("/d/" + id + "/" + 5 + "/", function(error, graph_data) {

            console.debug(error)
            console.debug(graph)

            graph.force
                .nodes(graph_data.nodes)
                .links(graph_data.links)
                .start();

            var link = graph.svg.selectAll(".link")
                .data(graph_data.links)
                .enter().append("line")
                .attr("class", "link");

            var node = graph.svg.selectAll(".node")
                .data(graph_data.nodes)
                .enter().append("circle")
                .attr("class", "node")
                .attr("r", function(n) {
                    console.debug(n);
                    return 10 - 1.5*n.level;
                })
                .call(graph.force.drag);

            graph.force.on("tick", function() {
            link.attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });

            node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
            });
        });
    }

    // Start without id
    return graph;
})
