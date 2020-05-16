
var commonModule = (function() {

    var datasets_raw = [];
    var datasets_meta = {};
    
    function set_datasets(ds_raw) {
        datasets_raw = ds_raw;
    }

    function get_datasets() {
        return datasets_raw;
    }

    function render_dropdown(drpData, drpElem) {

        // Erasing the previous content
        drpElem.html("");

        var fieldCount = drpData.length;

        if(fieldCount) {

            for(var i=0; i< fieldCount; i++) {
                var cur_field = drpData[i];

                if(cur_field != "_id") {
                    drpElem.append(getOptionHTML(cur_field));
                }  
            }
        } else {
            // @TODO: Display empty data message
        }

    }

    function getOptionHTML(option_name) {
        return "<option value='"+ option_name +"' >" + option_name + "</option>";
    }

    function get_cat_cols(col_map) {
        
        var cat_cols = []

        for(c in col_map) {

            // Check if the current column type is categorical
            // If yes then push the col
            if(col_map[c] == "Categorical") {
                cat_cols.push(c);
            }
        
        }

        return cat_cols;
    }

    function get_ds_meta() {
        return datasets_meta;
    }

    function set_ds_meta(d_meta) {
        datasets_meta = d_meta;
    }

    // All the functions which are accessible outside
    return {
        'set_datasets': set_datasets, // Raw datasets data
        'get_datasets': get_datasets,
        'set_ds_meta': set_ds_meta, // Formatted datasets data in to Objects
        'get_ds_meta': get_ds_meta, 
        'render_dropdown': render_dropdown, 
        'get_cat_cols': get_cat_cols, // Return the dataset fields of type Categorical
        'getOptionHTML': getOptionHTML
    }
  
})();