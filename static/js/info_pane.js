define(["ractive", "lib/underscore", "text!templates/info_pane.html"], function(R, _, pane_template) {

	////////////////////////////////////////
	//                                    //
	//            Ractor View             //
	//                                    //
	////////////////////////////////////////

	info_pane = {}

	info_pane.view = new R({
		template : pane_template,
		el : "info_pane",
		data : {
			id : undefined,
			title : undefined,
			abstract : undefined,
			paper : undefined,
			authors : undefined
		}
	});

	////////////////////////////////////////
	//                                    //
	//               Events               //
	//                                    //
	////////////////////////////////////////

	info_pane.events = function() {
		
		// When button is pressed
		info_pane.view.on("submit-click", function(e) {
			console.log(info_pane.view.data.id);
			info_pane.graph.update(info_pane.view.data.id);
		});

	}

	////////////////////////////////////////
	//                                    //
	//            Functions               //
	//                                    //
	////////////////////////////////////////

	info_pane.set_graph = function(graph) {
		info_pane.graph = graph;
	}

	info_pane.display = function(node) {
	    info_pane.view.set("id", node.id);
	    info_pane.view.set("title", node.title);
	    info_pane.view.set("abstract", node.abstract);
	    info_pane.view.set("authors", node.authors)
	    info_pane.view.set("paper", true);
	}

	////////////////////////////////////////
	//                                    //
	//               Return               //
	//                                    //
	////////////////////////////////////////

	//view.events();
	return info_pane;
});
