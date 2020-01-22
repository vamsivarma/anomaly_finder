$(document).ready(function() {
  $('[data-toggle="tooltip"]').tooltip();

  get_datasets();

  /*$("#ds-upload-button").click(function() { 
    var fd = new FormData(); 
    var files = $('#ds-upload-file')[0].files[0]; 
    fd.append('file', files); 

    $.ajax({ 
        url: '/mupload', 
        type: 'post', 
        data: fd, 
        contentType: false, 
        processData: false, 
        success: function(response){ 
            alert(response);
        }
    }); 
  });*/

});

function goHome() {
  $('#sign-out').modal('hide');
  window.location.href = "/";
}

function get_datasets() {

  $.ajax({ 
    url: '/datasets', 
    type: 'GET',
    dataType: "json", 
    success: function(response) { 
      render_datasets_table(response);

    },
    error: function(response) {
        console.log("Datasets API failed")
    } 
  }); 

}

// @TODO: Need to do better exception handling here
function render_datasets_table(res) {

  //var dataset_meta = JSON.parse(res);
  var row_data = res.rows;

  // Erasing the previous content
  var datasetTableBodyElem = $("#dataset-table-holder tbody");
  datasetTableBodyElem.html("");

  var rowCount = row_data.length;

  if(rowCount) {

    for(var i=0; i< rowCount; i++) {
      // Add a separate row for each dataset in the response
      datasetTableBodyElem.append(getRowHTML(i, row_data[i]));
    }
  } else {
    // @TODO: Display empty data message
  }
  register_table_events();
}

function getRowHTML(row_index, row_obj) {
  var rowHTML = "<tr><th>";
  rowHTML += row_index + "</th><td>";
  rowHTML += row_obj['name'] + "</td><td>";
  rowHTML += row_obj['size'] + "</td><td>";
  rowHTML += row_obj['col_count'] + "</td><td>";
  rowHTML += row_obj['row_count'] + "</td></tr>";

  return rowHTML;
}

function register_table_events() {

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




















