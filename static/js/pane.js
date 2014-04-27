define(["ractive", "text!templates/pane.html"], function(R, pane_template) {

	////////////////////////////////////////
	//                                    //
	//            Ractor View             //
	//                                    //
	////////////////////////////////////////

	var view = new R({
		template : pane_template,
		el : "pane",
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

	view.events = function() {
        // TODO
    }

	////////////////////////////////////////
	//                                    //
	//            Functions               //
	//                                    //
	////////////////////////////////////////

    view.graph_click = function(title, id) {
        view.set("id", id);
        view.set("title", title);
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
