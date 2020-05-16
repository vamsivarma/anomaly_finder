$(window).on('load', function() {
    datasetsModule.initModule()
});


var datasetsModule = (function() {
  
  var _this = '';
  var holderElem = '';
  var datasetTableBodyElem = '';

  function initModule() {
    _this = this;
    holderElem = $('#datasets-holder'); 
    datasetTableBodyElem = holderElem.find("#dataset-table-holder tbody");

    apiModule.get_datasets(render_datasets_table.bind(this))
  }

  // @TODO: Need to do better exception handling here
  function render_datasets_table(res) {

    //var dataset_meta = JSON.parse(res);
    var row_data = res.rows;

    var ds_meta = {};

    // Persist the datasets information to reuse in other pages
    commonModule.set_datasets(row_data);

    // Erasing the previous content
    datasetTableBodyElem.html("");

    var rowCount = row_data.length;

    if(rowCount) {

      for(var i=0; i< rowCount; i++) {

        var cur_ds_name = row_data[i]['name'];
        var cur_ds_columns = row_data[i]['columns']
        var cur_ds_col_type_map = row_data[i]['col_type_map']

        ds_meta[cur_ds_name] = {
            'columns': [],
            'col_type_map': {}
        }

        // Persisting the dataset information
        ds_meta[cur_ds_name]['columns'] = cur_ds_columns;
        ds_meta[cur_ds_name]['col_type_map'] = cur_ds_col_type_map;

        // Add a separate row for each dataset in the response
        datasetTableBodyElem.append(getRowHTML(i, row_data[i]));
      }

      // Persist the datasets meta information to reuse in other pages
      commonModule.set_ds_meta(ds_meta);

    } else {
      // @TODO: Display empty data message
    }

    register_table_events();

  }

  function getRowHTML(row_index, row_obj) {
    var rowHTML = "<tr><th>";
    rowHTML += (row_index + 1) + "</th><td>";
    rowHTML += row_obj['name'] + "</td><td>";
    rowHTML += row_obj['size'] + "</td><td>";
    rowHTML += row_obj['col_count'] + "</td><td>";
    rowHTML += row_obj['row_count'] + "</td></tr>";

    return rowHTML;
  }

  function register_table_events() {

      // @TODO: Need to close the context menu when clicked outside

      // Events for context menu on the data table
      $('#dataset-table-holder td').on('contextmenu', function(e) {
      
        var top = e.pageY - 10;
        var left = e.pageX - 10;
        
        $("#context-menu").css({
          display: "block",
          top: top,
          left: left
        });
        
        return false;
        
      }).on("click", function() {
      
        $("#context-menu").hide();
        
      });
      
      $("#context-menu a").on("click", function() {
        $(this).parent().hide();
      });

  }

  return {
    'initModule': initModule
  }

})();






















