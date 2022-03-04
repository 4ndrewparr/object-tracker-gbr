# custom tracker 2.0

class TrackedObject:
    
    def __init__(self, bbox, conf, frame_idx, img, tobj_idx, score_hit, score_nohit, score_tracked, life_max, life_min, num_moves, verbose):
        self.bboxes_orig = [bbox]
        self.bboxes = [bbox]
        self.confs = [conf]
        self.frames = [frame_idx]
        self.img = self.crop(bbox, img)  # cropped img of the cots
        self.tobj_idx = tobj_idx
        self.score_hit = score_hit
        self.score_nohit = score_nohit
        self.score_tracked = score_tracked
        self.life_max = life_max
        self.life_min = life_min
        self.num_moves = num_moves
        self.verbose = verbose
        self.life = 1
        self.moves = []
        self.hits = 1
        self.tracks = 0
        self.tracking = False
        
    def crop(self, bbox, img):
        """
        preds can have out of bounds values (including negative values)
        but slicing an image array from a negative value yields error
        i.e.: tobj-bbox: [-1, 507, 31, 558] img: (720, 1280, 3) -> tobj-img: (51, 0, 3) 
        """
        def pp(coord): return max(0, round(coord))
        return img[pp(bbox[1]):pp(bbox[3]),pp(bbox[0]):pp(bbox[2]),:]
    
    def get_move(self, bbox1, bbox2):
        """
        move vector between two boxes, from center to center
        bbox format expected x1y1x2y2
        """
        mid1 = (bbox1[0]+(bbox1[2]-bbox1[0])/2, bbox1[1]+(bbox1[3]-bbox1[1])/2)
        mid2 = (bbox2[0]+(bbox2[2]-bbox2[0])/2, bbox2[1]+(bbox2[3]-bbox2[1])/2)
        move = [mid2[0]-mid1[0], mid2[1]-mid1[1]]
        return move
    
    def make_inbound(self, bbox, img):
        """
        clips bbox to frame bounds
        returns list of scalars (np.clip returns scalar when only one element)
        """
        h, w = img.shape[:2]
        return [np.clip(bbox[0], 0, w), np.clip(bbox[1], 0, h), np.clip(bbox[2], 0, w), np.clip(bbox[3], 0, h)]
    
    def update(self, bbox, conf, frame_idx, img, hit=True, tracked=False):
        """
        tobj update in frame (new state predicted-by-model=hit, tracked or estimated)
        
        when updating a tobj with the prediction of the model, 
        we are assuming prediction accuracy > matchTemplate accuracy
        """
        self.bboxes_orig.append(bbox)
        if self.bboxes:  # existing previous bbox
            # add move
            move = self.get_move(self.bboxes[-1], bbox)
            self.moves.append(move)
            # cut bbox correspondingly if predicted to go oob next
            dif_x, dif_y = np.mean(self.moves[-self.num_moves:], axis=0)
            bb_est = [bbox[0]+dif_x, bbox[1]+dif_y, bbox[2]+dif_x, bbox[3]+dif_y]  # we move it to t+1
            bb_est = self.make_inbound(bb_est, img)  # we cut it with bounds
            bbox = [bb_est[0]-dif_x, bb_est[1]-dif_y, bb_est[2]-dif_x, bb_est[3]-dif_y]  # we move it back to t
        # update bbox, conf, img (hits, tracks and estimations)
        self.bboxes.append(bbox)
        self.confs.append(conf)
        self.img = self.crop(bbox, img)
        if self.verbose:
            print(f"TO#{self.tobj_idx:02} UPDATE bbox: {[round(i) for i in bbox]} img: {self.img.shape} {self.img.size}")
        # update frame, hits, span (hits and tracks)
        if hit:
            self.frames.append(frame_idx)  # only used for span
            self.hits += 1
            self.life += self.score_hit
        # update tracks
        if tracked: # (only tracks)
            self.tracks += 1
            self.life += self.score_tracked
        self.life = min(self.life, self.life_max)  # cap life at life_max
        if not self.tracking and self.life == self.life_min:  # reached life_min for the first time
            self.tracking = True
            self.life = self.life_max  # we set it at life_max to imitate norfair behaviour
        
    def __repr__(self):
        repr_str = f"TO#{self.tobj_idx:02}: " +\
        f"life={self.life}, last_frame={self.frames[-1]}, bbox={[round(i) for i in self.bboxes[-1]]}"
        return repr_str
        
class Tracker:
    
    def __init__(
        self, iou_thres=.50, match_thres=.45, resilience=3,
        dynamic=False, scales_num=5, scales_start=.90, scales_end=1., verbose=False,
        min_area=288, r_dist=2, max_tracks=5, fill_unmatched=False,
        min_conf=.0, num_moves=5,
        life_min=3, life_max=6, fn_conf=np.max, iou_start=.01,
        score_hit=1, score_nohit=-1, score_tracked=.5, r_dist_moves=.5,
        **kwargs,
    ):
        self.iou_thres = iou_thres
        self.match_thres = match_thres
        self.resilience = resilience
        self.r_dist = r_dist
        self.r_dist_moves = r_dist_moves
        self.dynamic = dynamic  # dynamic template matching (different scales)
        self.scales_num = scales_num
        self.scales_start = scales_start
        self.scales_end = scales_end
        self.verbose = verbose
        self.min_area = min_area
        self.min_conf = min_conf
        self.max_tracks = max_tracks
        self.fill_unmatched = fill_unmatched
        self.life_min = life_min
        self.life_max = life_max
        self.score_hit = score_hit
        self.score_nohit = score_nohit
        self.score_tracked = score_tracked
        self.fn_conf = fn_conf
        self.iou_start = iou_start
        self.num_moves = num_moves
        self.tobjs = []
        self.img = None  # cv2 BGR
        self.frame_idx = -1
        self.total_tobjs = 0  # used to assign idx to tobj
        self.bboxes_garbage = []
    
    def area_of_interest(self, tobj):
        """
        returns cropped area of frame where we will look for matches
        negative coords slices yield error, all negative coords must be converted to 0    
        
        last bbox can be based on estimation of next location of tobj, so it can be of size=0
        like bb_ref=[0.0, 25.46, 0.15, 63.87] with times=.5 becomes img_cropped with shape=(77, 0, 3)
        """
        if tobj.moves:
            bb_ref = self.estimate(tobj)
            times = self.r_dist_moves
            #times = self.r_dist * (self.frame_idx-tobj.frames[-1])
        else:
            bb_ref = tobj.bboxes[-1]
            times = self.r_dist
            #times = self.r_dist * (self.frame_idx-tobj.frames[-1])

        w = bb_ref[2]-bb_ref[0]
        h = bb_ref[3]-bb_ref[1]
        min_row = round(max(bb_ref[1]-h*times, 0))  # we cannot have negative values when slicing
        min_col = round(max(bb_ref[0]-w*times, 0))
        max_row = round(max(bb_ref[3]+h*times, 0))
        max_col = round(max(bb_ref[2]+w*times, 0))
        img_cropped = self.img[  # we crop a square around bb_ref with times*w/h margin
            min_row: max_row,
            min_col: max_col
        ]
        #print(f"{[min_row, min_col, max_row, max_col]}")
        return img_cropped, min_col, min_row
    
    def get_match_results(self, tobj, img_cropped, scale, min_col, min_row):
        """           
        # according to cv2 documentation:
        #To shrink an image, it will generally look best with INTER_AREA interpolation,
        #whereas to enlarge an image, it will generally look best with c::INTER_CUBIC (slow)
        #or INTER_LINEAR (faster but still looks OK).
        """
        if scale > 1.:
            inter = cv2.INTER_CUBIC
        else:  # with scale==1. any inter does nothing
            inter = cv2.INTER_AREA

        tobj_scaled = cv2.resize(
            tobj.img, None, fx=scale, fy=scale,
            interpolation=inter,
        )

        #print(img_cropped.shape, tobj_scaled.shape)
        matches = cv2.matchTemplate(img_cropped, tobj_scaled, cv2.TM_CCOEFF_NORMED)

        _, max_value, _, max_loc = cv2.minMaxLoc(matches)  # min_value, max_value, min_loc, max_loc (locs are upper-left corner x,y tuples)
        max_loc = (max_loc[0]+min_col, max_loc[1]+min_row)  # we need to convert match coords to whole img coords
        h, w = tobj_scaled.shape[:2]
        bbox_matched = [max_loc[0], max_loc[1], max_loc[0]+w, max_loc[1]+h]
        
        return max_value, scale, bbox_matched
    
    def match2img(self, tobj, img_cropped, min_col, min_row):
        """
        matches tobj to frame img
        """
        #img_cropped, min_col, min_row = self.area_of_interest(tobj)
        
        if self.dynamic:
            scales = np.linspace(self.scales_start, self.scales_end, self.scales_num)
        else:
            scales = [1.]
            
        results = []
        for scale in scales:
            results += [self.get_match_results(tobj, img_cropped, scale, min_col, min_row)]   
            #results += [(max_value, scale, bbox_matched)]

        max_value, scale, bbox_matched = sorted(results, key=lambda x: x[0], reverse=True)[0]
        return max_value, scale, bbox_matched, results
    
    def match2tobj(self, bbox):
        """
        returns best match (best iou) between predicted bbox
        and matchTemplate of tobj-bbox in current image for each existing tobj
        """
        # get best template match in current image for each tobj
        bboxes_matched = []
        tobjs_matched = []
        for t, tobj in enumerate(self.tobjs):
            
            if tobj.img.size == 0: continue  # like when tobj estimated goes oob
                
            img_cropped, min_col, min_row = self.area_of_interest(tobj)
            if img_cropped.size == 0: continue  # can happen if width of tobj bbox is less than one
                
            max_value, scale, bbox_matched, results = self.match2img(tobj, img_cropped, min_col, min_row)
            bboxes_matched += [bbox_matched]
            tobjs_matched += [t]

        if bboxes_matched:
            # for each bbox pred, get best iou
            ious = box_iou(  # expects (x1y1x2y2)
                torch.tensor(bbox).unsqueeze(0),
                torch.tensor(bboxes_matched),
            )  # returns 1xm matrix with ious
            iou = torch.max(ious[0]).item()
            idx = torch.argmax(ious[0]).item()
            idx = tobjs_matched[idx]
        else:
            iou, idx = 0, -1  # idx doesnt matter in this case
        return iou, idx
        
    def update(self, bboxes, confs, frame_idx, img):
        
        # assign img and frame
        self.img = img
        self.frame_idx = frame_idx
        
        # update tobjs
        for i, bbox in enumerate(bboxes):
            iou, idx = self.match2tobj(bbox)

            if iou > 0:
                if self.verbose:
                    print(f"new pred bbox: {[round(i) for i in bbox]}")
                    print(f"tobj matched: {self.tobjs[idx]}")
                
                # bbox that hits
                if self.tobjs[idx].moves:  # it was matched with estimate (we can ask for more iou)
                    min_iou = self.iou_thres
                else:  # it was matched with previous location
                    min_iou = self.iou_start
                if iou >= min_iou:

                    # update attrs of tobj matched (the bbox/obj detected is being tracked already)
                    self.tobjs[idx].update(bbox, confs[i], frame_idx, img, hit=True, tracked=False)

                    if self.verbose:
                        print(
                            f"id: {self.tobjs[idx].tobj_idx:02} iou: {iou:.2f} " +
                            f"bbox: {[round(i) for i in self.tobjs[idx].bboxes[-2]]}->{[round(i) for i in bbox]}"
                        )
                        
                    continue
            #else:
            # instantiate new tobj
            new_tobj = TrackedObject(
                bbox, confs[i], frame_idx, img,
                tobj_idx=self.total_tobjs, score_hit=self.score_hit,
                score_nohit=self.score_nohit, score_tracked=self.score_tracked,
                life_max=self.life_max, life_min=self.life_min, num_moves=self.num_moves,
                verbose=self.verbose,
            )
            self.total_tobjs += 1
            self.tobjs += [new_tobj]
        
        # update other tobjs and remove dead tobjs
        ids_keep = []
        ids_garbage = []
        for t, tobj in enumerate(self.tobjs):
            # update life of those tobjs not hit
            if tobj.frames[-1] != self.frame_idx:
                tobj.life += self.score_nohit
            # mark alive tobjs
            if not tobj.tracking and tobj.life < 0:  # they never initialized
                ids_garbage += [t]
                continue
            if tobj.tracking and tobj.life < self.life_min - self.resilience:  # life less than 0 can be ok for tracked tobjs if resilience
                continue
            ids_keep += [t]
        # get garbage preds before removing
        for t, tobj in enumerate(self.tobjs):
            if t in ids_garbage:
                self.bboxes_garbage += tobj.bboxes_orig
        # remove dead tobjs  
        self.tobjs = np.array(self.tobjs)[ids_keep].tolist()
        if self.verbose:
            for tobj in self.tobjs: print(tobj)
        
    def get_area(self, bbox):
        """
        area of bbox
        bbox format expected x1y1x2y2
        """
        return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                
    def get_distance(self, bbox1, bbox2):
        """
        l2 norm measured on bboxes' middle points
        bbox format expected x1y1x2y2
        """
        mid1 = (bbox1[0]+(bbox1[2]-bbox1[0])/2, bbox1[1]+(bbox1[3]-bbox1[1])/2)
        mid2 = (bbox2[0]+(bbox2[2]-bbox2[0])/2, bbox2[1]+(bbox2[3]-bbox2[1])/2)
        dist = ((mid1[0]-mid2[0])**2 + (mid1[1]-mid2[1])**2)**.5
        return dist
    
    def speed(self):
        """
        only used to get movement statistics
        we just need to iterate over train set and tracker_.update() using gt bboxes
        """
        moves = []
        for tobj in self.tobjs:
            if len(tobj.bboxes) > 1:
                abs_move = self.get_distance(tobj.bboxes[-1], tobj.bboxes[-2])
                rel_move = abs_move/(self.get_area(tobj.bboxes[-2])**.5)
                moves += [(abs_move, rel_move, self.frame_idx)]
        return moves
    
    def make_inbound(self, bbox):
        """
        clips bbox to frame bounds
        returns list of scalars (np.clip returns scalar when only one element)
        """
        h, w = self.img.shape[:2]
        return [np.clip(bbox[0], 0, w), np.clip(bbox[1], 0, h), np.clip(bbox[2], 0, w), np.clip(bbox[3], 0, h)]
    
    def estimate(self, tobj):
        """
        estimates position of a tobj in frame
        """
        dif_x, dif_y = np.mean(tobj.moves[-self.num_moves:], axis=0)
        bb = tobj.bboxes[-1]
        bb_est = [bb[0]+dif_x, bb[1]+dif_y, bb[2]+dif_x, bb[3]+dif_y]  # we keep same dims of last bbox
        bb_est = self.make_inbound(bb_est)
        return bb_est
    
    def find(self):
        """
        updates unmatched tobjs with track or estimate
        """
        bbs_tr = []
        scores_tr = []
        for tobj in self.tobjs:
            
            # filter tobjs
            if tobj.img.size == 0: continue  # if estimated to go oob, img has size 0
            if tobj.frames[-1] == self.frame_idx: continue  # not predicted in current frame
            if tobj.life < self.life_min: continue
            if tobj.tracks > self.max_tracks: continue  # has not been tracked more than max_tracks times
            if self.fn_conf(tobj.confs) < self.min_conf: continue
                
            img_cropped, min_col, min_row = self.area_of_interest(tobj)
            if img_cropped.size == 0: continue  # can happen if width of tobj bbox is less than one
            max_value, scale, bbox_matched, results = self.match2img(tobj, img_cropped, min_col, min_row)
            
            dist = self.get_distance(bbox_matched, tobj.bboxes[-1])
            area = self.get_area(bbox_matched)
            if self.verbose:
                print(
                    f"-> TO#{tobj.tobj_idx:02} undetected tobj found " + 
                    f"(best_match={max_value:.3f}, scale={scale:.3f}, dist={dist:.1f})"
                )
            
            # no good match found for tobj (we will estimate bbox)
            if (
                max_value < self.match_thres
                or area < self.min_area
            ):
                if self.fill_unmatched:
                    
                    bb_est = self.estimate(tobj)
                    if self.verbose: print(f"{[i for i in bb_est]}")
                    score_est = tobj.confs[-1]

                    if self.get_area(bb_est)>0:
                        if self.verbose: print(f"tobj unmatched filled with estimate")
                        bbs_tr += [bb_est]
                        scores_tr += [score_est]
                    
                        # partial tobj update
                        tobj.update(bb_est, score_est, frame_idx=None, img=self.img, hit=False, tracked=True)
                
                else:
                    continue
                
            # good match found for tobj
            else:
                bb_tr = bbox_matched
                
                score_tr = tobj.confs[-1]  # lets just assign the last prediction's conf
                
                bbs_tr += [bb_tr]
                scores_tr += [score_tr]
                
                # update tobj
                tobj.update(bb_tr, score_tr, self.frame_idx, self.img, hit=False, tracked=True)
                if self.verbose:
                    print(f"-> TOBJ ADDED (bb={[round(i) for i in bb_tr]}, score={score_tr:.2f})")
                
        return bbs_tr, scores_tr