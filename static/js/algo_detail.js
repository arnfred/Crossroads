define(["ractive", "lib/underscore", "text!templates/algo_detail.html"], function(R, _, pane_template) {

	////////////////////////////////////////
	//                                    //
	//            Ractor View             //
	//                                    //
	////////////////////////////////////////

	var view = new R({
		template : pane_template,
		el : "algo_detail",
		data : {
			hide: true,
            neighbors_list: undefined
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

	view.display = function(n) {
		view.set("hide", false);
	    view.set("neighbors_list", n);
	}

	////////////////////////////////////////
	//                                    //
	//               Return               //
	//                                    //
	////////////////////////////////////////

	//view.events();
	return view;
});
