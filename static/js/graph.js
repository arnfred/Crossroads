define(["lib/d3.v3.min"], function(d3) {

    var graph = {};

    var width = parseInt(d3.select("#graph").style("width"));
        height = parseInt(d3.select("#graph").style("height")) - 10;

    graph.svg = d3.select("#graph").append("svg")
        .attr("width", width)
        .attr("height", height);

    graph.force = d3.layout.force()
        .gravity(.15)
        .distance(10)
        .charge(-150)
        .linkDistance(function (l) { return 100*l.value; })
        .linkStrength(function (l) { return l.value; })
        .size([width, height]);

    // Add data to graph
    graph.start = function(id, click_fun) {

        if (id == "") {
            id = "1304.5220"
        }

        // Load data and start force layout
        d3.json("/d/" + id + "/" + 5 + "/", function(error, graph_data) {

            // Stop and clear existing force layout
            graph.svg.selectAll(".link").remove()
            graph.svg.selectAll(".node").remove()

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
                .on("mouseover", function(n) {
                    setTimeout(function() {
                        click_fun(n.title, n.abstract, n.authors, n.id);
                    }, 200);
                })
                .on("dblclick", function(n) {
                    graph.start(n.id, click_fun)
                })
                .attr("r", function(n) {
                    return 13 - 3*n.level;
                })
                .attr("fill", function(n) {
                    if (n.id == id) {
                        return  HSVtoHEX(300, 100, 80)
                    }
                    else {
                        return HSVtoHEX(210, 100, 100 - 30 * n.level)
                    }
                })
                .attr("stroke", function(n) {
                    if (n.id == id) {
                        return  HSVtoHEX(300, 100, 60)
                    }
                    else {
                        return HSVtoHEX(210, 100, 120 - 30 * n.level)
                    }
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



function HSVtoHEX(h,s,v) { 
    var rgb = HSVtoRGB(h,s,v)
    return RGBtoHEX(rgb.r, rgb.g, rgb.b)
}

function RGBtoHEX(r,g,b) {
    if (r && g === undefined && b === undefined) {
        g = r.g, b = r.g, r = r.r;
    }
    return "#"+toHex(r)+toHex(g)+toHex(b)
}

function toHex(n) {
    n = parseInt(n,10);
    if (isNaN(n)) return "00";
    n = Math.max(0,Math.min(n,255));
    return "0123456789ABCDEF".charAt((n-n%16)/16)
        + "0123456789ABCDEF".charAt(n%16);
}

function HSVtoRGB(h, s, v) {
    var r, g, b;
    if (h && s === undefined && v === undefined) {
        s = h.s, v = h.v, h = h.h;
    }

    h = h / 360;
    s = s / 100;
    v = v / 100;

    var i = Math.floor(h * 6);
    var f = h * 6 - i;
    var p = v * (1 - s);
    var q = v * (1 - f * s);
    var t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }
    return { 'r':Math.round(r * 255), 'g':Math.round(g * 255), 'b':Math.round(b * 255)};
}