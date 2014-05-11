define(["lib/d3.v3.min", "ractive", "text!templates/search.html"],
       function(d3, R, search_template) {

    ////////////////////////////////////////
    //                                    //
    //            Ractor View             //
    //                                    //
    ////////////////////////////////////////

    var search = {}

    search.view = new R({
	    template : search_template,
	    el : "search",
	    data : {
		terms : undefined,
		is_shown : true,
		results : undefined
	    }
    });

    ////////////////////////////////////////
    //                                    //
    //               Events               //
    //                                    //
    ////////////////////////////////////////

    search.events = function() {

	// When button is pressed
	search.view.on("submit-click", do_search)

	// When search result is clicked
	search.view.on("goto-graph", function(e) {
	    init_graph(e.context.id);
	});

	// Listen for enter key
	document.onkeypress = function (e) {
	    if (e.charCode == 13) do_search()
	}
    }

    ////////////////////////////////////////
    //                                    //
    //            Functions               //
    //                                    //
    ////////////////////////////////////////


    // Set graph
    search.set_graph = function(graph) {
	search.graph = graph;
    }


    var do_search = function() {
	var input_text = search.view.get("terms");
	var url_terms = encodeURIComponent(input_text);
	// Get result
        d3.json("/search/" + url_terms, function(error, search_data) {
	    // If search result has only one paper, then go directly to graph
	    if (search_data.length == 1) {
		init_graph(search_data[0].id);
	    }

	    // Otherwise show search results
	    else {
		show_results(search_data);
	    }
	});
    }

    var init_graph = function(paper_id) {
	// Fade out search and initialize graph ...
	search.graph.update(paper_id);
	search.view.set("is_shown", undefined);
    }


    var show_results = function(search_data) {

	search.view.set("results", search_data);
    }



    ////////////////////////////////////////
    //                                    //
    //               Return               //
    //                                    //
    ////////////////////////////////////////

    search.events();
    return search;
});
