define(["ractive", "lib/underscore", "text!templates/info_pane.html"], function(R, _, pane_template) {

	////////////////////////////////////////
	//                                    //
	//            Ractor View             //
	//                                    //
	////////////////////////////////////////

	var view = new R({
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


	////////////////////////////////////////
	//                                    //
	//            Functions               //
	//                                    //
	////////////////////////////////////////

    view.display = function(node) {
		view.set("id", node.id);
		view.set("title", node.title);
		view.set("abstract", node.abstract);
		view.set("authors", node.authors)
		view.set("paper", true);
    }

	////////////////////////////////////////
	//                                    //
	//               Return               //
	//                                    //
	////////////////////////////////////////

	view.events();
	return view;
});
