define(["lib/d3.v3.min", "lib/jquery", "ractive", "text!templates/search.html"],
	   function(d3, $, R, search_template) {

	////////////////////////////////////////
	//                                    //
	//            Ractor View             //
	//                                    //
	////////////////////////////////////////

	var search = {}

	var perPage = 5;
	var numItems = undefined;
	var numPages = undefined;
	var currPage = 0;
	var data = undefined;

	search.view = new R({
		template : search_template,
		el : "search-page",
		data : {
			terms : undefined,
			is_shown : true,
			no_result : false,
			results_shown : false,
			results : undefined,
			length : undefined,
			duration : undefined,
			is_first_page : true,
			is_last_page : true,
			currPage : 1,
			numPages : 1
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

		// When First button is clicked
		search.view.on("first", function(e) {
			if (!search.view.get("is_first_page")) {
				goTo(0);
			}
		});
		// When Previous button is clicked
		search.view.on("previous", function(e) {
			if (!search.view.get("is_first_page")) {
				previous();
			}
		});
		// When Next button is clicked
		search.view.on("next", function(e) {
			if (!search.view.get("is_last_page")) {
				next();
			}
		});
		// When Last button is clicked
		search.view.on("last", function(e) {
			if (!search.view.get("is_last_page")) {
				goTo(numPages-1);
			}
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
		d3.json("/search/" + url_terms, function(error, search_result) {
			data = search_result.data;

			// If search result has no paper
			if (search_result.data.length == 0) {
				search.view.set("no_result", true);
				search.view.set("results_shown", false);
				search.view.set("results", undefined);
			}
			else {
				search.view.set("no_result", false);

				// If search result has only one paper, then go directly to graph
				if (search_result.data.length == 1) {
					init_graph(search_result.data[0].id);
				}

				// Otherwise show search results
				if (search_result.data.length > 1) {
					show_results(search_result);
				}
			}
		});
	}

	var init_graph = function(paper_id) {
		// Fade out search and initialize graph ...
		search.graph.update(paper_id);
		search.view.set("is_shown", undefined);
	}


	var show_results = function(search_result) {
		numItems = search_result.data.length;
		numPages = Math.ceil(numItems/perPage);
		currPage = 0;

		search.view.set("length", data.length);
		search.view.set("duration", search_result.duration);
		search.view.set("results_shown", true);
		search.view.set("is_first_page", true);
		search.view.set("is_last_page", currPage == numPages-1);
		search.view.set("currPage", 1);
		search.view.set("numPages", numPages);

		goTo(currPage);
	}

	function previous(){
		goTo(currPage-1);
	}

	function next(){
		goTo(currPage+1);
	}

	function goTo(page){
		currPage = page;
		search.view.set("is_first_page", currPage == 0);
		search.view.set("is_last_page", currPage == numPages-1);
		search.view.set("results", data.slice(currPage*perPage,(currPage+1)*perPage));
	}


	////////////////////////////////////////
	//                                    //
	//               Return               //
	//                                    //
	////////////////////////////////////////

	search.events();
	return search;
});
