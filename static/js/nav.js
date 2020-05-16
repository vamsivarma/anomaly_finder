$(window).on('load', function() {
    navModule.initModule();
});

var navModule = (function() {
    var mainElem = '';
    var signOutElem = '';
    
    function initModule() {
        mainElem = $('#main-holder-elem');
        signOutElem =  mainElem.find('#sign-out');
        
        register_module_events();
    }

    function register_module_events() {
      
        mainElem.find('#nav-holder > li > a').on('click', function(){
          //$('.navbar-collapse').collapse('hide');
          var togglerElem = mainElem.find('.navbar-toggler');
      
          if(togglerElem.css('display') !== 'none') {
            // We reach here if the screen is small and toggler is visible
            togglerElem.click();
          }
      
        });
      }
      
      function goHome() {
        $('#sign-out').modal('hide');
        window.location.href = "/";
      }


    return {
      'initModule': initModule,
      'goHome': goHome
    }
  
  })();